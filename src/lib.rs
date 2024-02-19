use bitstream::BitStackWriter;

pub mod bitstream;
pub mod fse;
pub mod histogram;

pub use histogram::{Histogram, NormHistogram};

const TABLE_LOG_MIN: u32 = 5;
const TABLE_LOG_MAX: u32 = 15;
const TABLE_LOG_RANGE: std::ops::RangeInclusive<u32> = TABLE_LOG_MIN..=TABLE_LOG_MAX;
const TABLE_LOG_DEFAULT: u32 = 11;

#[allow(dead_code)]
const MASK: [usize; 33] = [
    0,
    0x1,
    0x3,
    0x7,
    0xF,
    0x1F,
    0x3F,
    0x7F,
    0xFF,
    0x1FF,
    0x3FF,
    0x7FF,
    0xFFF,
    0x1FFF,
    0x3FFF,
    0x7FFF,
    0xFFFF,
    0x1_FFFF,
    0x3_FFFF,
    0x7_FFFF,
    0xF_FFFF,
    0x1F_FFFF,
    0x3F_FFFF,
    0x7F_FFFF,
    0xFF_FFFF,
    0x1FF_FFFF,
    0x3FF_FFFF,
    0x7FF_FFFF,
    0xFFF_FFFF,
    0x1FFF_FFFF,
    0x3FFF_FFFF,
    0x7FFF_FFFF,
    0xFFFF_FFFF,
];

// Fast Mask lookup. It's quicker to read in the mask value than it is to
// calculate it with a shift & subtract.
#[inline]
fn find_mask(val: usize) -> usize {
    debug_assert!(val <= 32, "Masking shouldn't be above 32 bits at any point");
    unsafe { *MASK.get_unchecked(val) }
}

/// A wrapped byte vector for quickly pushing individual bytes into it.
struct DeferByteVec<'a> {
    v: &'a mut Vec<u8>,
    ptr: *mut u8,
    end: *mut u8,
}

impl<'a> DeferByteVec<'a> {

    /// Wrap a byte vector for quicker push operations.
    fn new(v: &'a mut Vec<u8>) -> Self {
        let spare = v.spare_capacity_mut().as_mut_ptr_range();
        Self {
            v,
            ptr: spare.start as *mut u8,
            end: spare.end as *mut u8,
        }
    }

    /// Reserve additional space in the byte vector, as needed.
    #[cold]
    fn reserve(&mut self, additional: usize) {
        unsafe {
            self.v
                .set_len(self.ptr.offset_from(self.v.as_ptr()) as usize);
        }
        self.v.reserve(additional);
        let spare = self.v.spare_capacity_mut().as_mut_ptr_range();
        self.ptr = spare.start as *mut u8;
        self.end = spare.end as *mut u8;
    }

    /// Push an additional byte onto the byte vector
    fn push(&mut self, val: u8) {
        if self.ptr == self.end {
            self.reserve(1);
        }
        unsafe {
            self.ptr.write(val);
            self.ptr = self.ptr.add(1);
        }
    }
}

impl<'a> Drop for DeferByteVec<'a> {
    fn drop(&mut self) {
        // Complete the update of the byte vector before we release it.
        unsafe {
            self.v
                .set_len(self.ptr.offset_from(self.v.as_ptr()) as usize);
        }
    }
}

pub fn fse_compress(src: &[u8], dst: &mut Vec<u8>) -> (NormHistogram, usize) {
    // Construct the histogram and write it out
    let hist = NormHistogram::new(src);
    hist.write(dst);

    // Encode the symbol stream
    let mut writer = BitStackWriter::new(dst);
    let fse_table = fse::EncodeTable::new(&hist);
    let mut src_iter = src.chunks(2).rev();
    let first = src_iter.next().unwrap();
    let first_byte = first.last().unwrap();
    let mut encode = fse::Encoder::new_first_symbol(&fse_table, *first_byte);
    if first.len() > 1 {
        encode.encode(&mut writer, *first.first().unwrap());
    }
    for n in src_iter {
        unsafe {
            let n1 = *n.get_unchecked(1);
            let n0 = *n.get_unchecked(0);
            encode.encode_raw(&mut writer, n1);
            if usize::BITS < 64 {
                writer.flush();
            }
            encode.encode_raw(&mut writer, n0);
        }
        writer.flush();
    }
    encode.finish(&mut writer);
    // Marker bit, so the decoder knows where the final bit in the stream is.
    writer.write_bits(1, 1);
    (hist, writer.finish())
}

/// Compress with two streams at once
pub fn fse_compress2(src: &[u8], dst: &mut Vec<u8>) -> usize {
    // Construct the shared histogram and write it out
    let hist = histogram::NormHistogram::new(src);
    hist.write(dst);

    let mut writer = BitStackWriter::new(dst);
    let fse_table = fse::EncodeTable::new(&hist);
    let mut src_iter = src.chunks(2).rev();
    let first = src_iter.next().unwrap();
    let (mut encode0, mut encode1) = if first.len() == 1 {
        let next = src_iter.next().unwrap();
        let mut encode0 = fse::Encoder::new_first_symbol(&fse_table, first[0]);
        let encode1 = fse::Encoder::new_first_symbol(&fse_table, next[1]);
        encode0.encode(&mut writer, next[0]);
        (encode0, encode1)
    } else {
        let encode0 = fse::Encoder::new_first_symbol(&fse_table, first[0]);
        let encode1 = fse::Encoder::new_first_symbol(&fse_table, first[1]);
        (encode0, encode1)
    };

    for n in src_iter {
        unsafe {
            encode1.encode_raw(&mut writer, *n.get_unchecked(1));
            if usize::BITS < 64 {
                writer.flush();
            }
            encode0.encode_raw(&mut writer, *n.get_unchecked(0));
            writer.flush();
        }
    }

    encode1.finish(&mut writer);
    encode0.finish(&mut writer);
    // Marker bit, so the decoder knows where the final bit in the stream is.
    writer.write_bits(1, 1);
    writer.finish()
}

// Run the decompressor on a stream compressed with `fse_compress`, returning
// how many bytes were added to the output stream.
pub fn fse_decompress(src: &[u8], dst: &mut Vec<u8>) -> Option<usize> {
    let len = dst.len();

    // Recover the histogram
    let (hist, src) = NormHistogram::read(src).ok()?;
    let mut reader = bitstream::BitStackReader::new(src)?;
    let mut fast_dst = DeferByteVec::new(dst);

    // Decode the stream
    let fse_table = fse::DecodeTable::new(&hist);
    let mut decode = fse::Decoder::new(&fse_table, &mut reader).unwrap();
    while let Some(s) = decode.decode_symbol(&mut reader) {
        fast_dst.push(s);
        if usize::BITS >= 64 {
            if let Some(s) = unsafe { decode.decode_symbol_no_reload(&mut reader) } {
                fast_dst.push(s);
            } else {
                break;
            }
        }
    }
    fast_dst.push(decode.finish());
    drop(fast_dst);
    Some(dst.len() - len)
}

// Run the decompressor on a stream compressed with `fse_compress2`, returning
// how many bytes were added to the output stream.
pub fn fse_decompress2(src: &[u8], dst: &mut Vec<u8>) -> Option<usize> {
    let len = dst.len();

    // Recover the histogram
    let (hist, src) = NormHistogram::read(src).ok()?;

    // Decode the stream
    let mut reader = bitstream::BitStackReader::new(src)?;
    let fse_table = fse::DecodeTable::new(&hist);
    let mut decode0 = fse::Decoder::new(&fse_table, &mut reader).unwrap();
    let mut decode1 = fse::Decoder::new(&fse_table, &mut reader).unwrap();
    let mut fast_dst = DeferByteVec::new(dst);
    unsafe {
        while let Some(s) = decode0.decode_symbol_no_reload(&mut reader) {
            fast_dst.push(s);
            if usize::BITS < 64 {
                reader.reload();
            }
            if let Some(s) = decode1.decode_symbol(&mut reader) {
                fast_dst.push(s);
            } else {
                fast_dst.push(decode1.finish());
                fast_dst.push(decode0.finish());
                drop(fast_dst);
                return Some(dst.len() - len);
            }
        }
        fast_dst.push(decode0.finish());
        fast_dst.push(decode1.finish());
    }

    drop(fast_dst);
    Some(dst.len() - len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn gen_sequence(prob: f64, size: usize) -> Vec<u8> {
        const LUT_SIZE: usize = 4096;
        let mut lut = [0u8; 4096];
        let prob = prob.clamp(0.005, 0.995);
        let mut remaining = LUT_SIZE;
        let mut idx = 0;
        let mut s = 0u8;
        while remaining > 0 {
            let n = ((remaining as f64 * prob) as usize).max(1);
            for _ in 0..n {
                lut[idx] = s;
                idx += 1;
            }
            s += 1;
            remaining -= n;
        }
        let mut out = Vec::with_capacity(size);
        let mut rng = rand::thread_rng();
        for _ in 0..size {
            let i: u16 = rng.gen();
            out.push(lut[i as usize & (LUT_SIZE - 1)]);
        }
        out
    }

    #[test]
    fn compress() {
        let src = gen_sequence(0.2, 1 << 16);
        let mut dst = Vec::with_capacity(src.len());
        let mut dec = Vec::with_capacity(src.len());
        fse_compress(&src, &mut dst);
        fse_decompress(&dst, &mut dec);
        if src != dec {
            panic!("Source and decoded result are different");
        }
    }

    #[test]
    fn compress2() {
        let src = gen_sequence(0.2, 1 << 16);
        let mut dst = Vec::with_capacity(src.len());
        let mut dec = Vec::with_capacity(src.len());
        fse_compress2(&src, &mut dst);
        fse_decompress2(&dst, &mut dec);
        if src != dec {
            panic!("Source and decoded result are different");
        }
    }
}
