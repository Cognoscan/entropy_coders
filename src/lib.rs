use bitstream::BitStackWriter;

pub mod bitstream;
pub mod histogram;
pub mod fse;

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

/// Run the compressor directly
pub fn fse_compress(src: &[u8], dst: &mut Vec<u8>) -> usize {
    let mut writer = BitStackWriter::new(dst);
    let hist = histogram::NormHistogram::new(src);
    let fse_table = fse::FseEncodeTable::new(&hist);
    let mut src_iter = src.iter().rev();
    let first = src_iter.next().unwrap();
    let mut encode = fse::FseEncode::new_first_symbol(&fse_table, *first);
    for n in src_iter {
        unsafe {
            encode.encode_raw(&mut writer, *n);
            if usize::BITS < 64 { writer.flush(); }
            encode.encode_raw(&mut writer, *n);
        }
        writer.flush();
    }
    encode.finish(&mut writer);
    // Marker bit, so the decoder knows where the final bit in the stream is.
    writer.write_bits(1, 1);
    writer.finish()
}

/// Compress with two streams at once
pub fn fse_compress2(src: &[u8], dst: &mut Vec<u8>) -> usize {
    let mut writer = BitStackWriter::new(dst);
    let hist = histogram::NormHistogram::new(src);
    let fse_table = fse::FseEncodeTable::new(&hist);
    let mut src_iter = src.chunks(2).rev();
    let first = src_iter.next().unwrap();
    let (mut encode0, mut encode1) = if first.len() == 1 {
        let next = src_iter.next().unwrap();
        let mut encode0 = fse::FseEncode::new_first_symbol(&fse_table, first[0]);
        let encode1 = fse::FseEncode::new_first_symbol(&fse_table, next[1]);
        encode0.encode(&mut writer, next[0]);
        (encode0, encode1)
    } else {
        let encode0 = fse::FseEncode::new_first_symbol(&fse_table, first[0]);
        let encode1 = fse::FseEncode::new_first_symbol(&fse_table, first[1]);
        (encode0, encode1)
    };

    for n in src_iter {
        unsafe {
          encode0.encode_raw(&mut writer, *n.get_unchecked(0));
          if usize::BITS < 64 { writer.flush(); }
          encode1.encode_raw(&mut writer, *n.get_unchecked(1));
          writer.flush();
        }
    }

    encode0.finish(&mut writer);
    encode1.finish(&mut writer);
    // Marker bit, so the decoder knows where the final bit in the stream is.
    writer.write_bits(1, 1);
    writer.finish()
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
        let src = gen_sequence(0.2, 1 << 15);
        let mut dst = Vec::with_capacity(src.len());
        fse_compress(&src, &mut dst);
    }
}
