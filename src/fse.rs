/*!
# Finite State Entropy (tANS) Implementation

These components provide the fundamental tools for implementing a tANS encoder
and decoder, referred to here as Finite State Entropy. These are specifically
for replicating the encoding/decoding scheme used by `zstd`, though it certainly
can be used in other compression schemes.

## Usage: Compression

First, a [normalized histogram][crate::histogram::NormHistogram] must be built
up for use with the entropy coder. Then, an [`EncodeTable`] can be constructed
from it. This table can be used with one or more [`Encoder`] units to compress
the data used to create the histogram.

Encoders can be interleaved or used alongside additional bits, as each one only
mutably borrows the [`BitStackWriter`] when encoding a symbol.

Fine-grained `unsafe` functions for encoding a symbol without flushing bits out
of the writer are available for achieving the fastest possible encoding speeds,
at the cost of additional manual reasoning about the maximum number of bits per
encoding operation.

Here's an example flow, using a single encoder. The output is finished with a
"1" marker bit in order for the decoder to identify the final bit in the byte
stream (same as `zstd`). Note that the histogram is not encoded in this
sequence; it must either be encoded before the main encoding sequence, or be
derived separately.

```
# use entropy_coders::fse::*;
# use entropy_coders::histogram::*;
# use entropy_coders::bitstream::*;
# let data = (0u8..=255u8).collect::<Vec<u8>>();

// Given `data` as the source Vec<u8>, construct the encoding table
let hist = NormHistogram::new(&data);
let table = EncodeTable::new(&hist);

// Prepare the encoder and implicitly encode the first byte. Because the
// bitstream must be read in reverse order, we reverse *now* so that the decoder
// outputs bytes in the same order as the original data vector.
let mut iter = data.iter().rev();
// Note: In a real encoder, zero-sized data should be caught before this point.
let first = iter.next().unwrap();
let mut encoder = Encoder::new_first_symbol(&table, *first);

let mut output = Vec::with_capacity(data.len());
let mut writer = BitStackWriter::new(&mut output);
for n in iter {
    encoder.encode(&mut writer, *n);
}
encoder.finish(&mut writer);
writer.write_bits(1, 1);
writer.finish();
```

*/


use crate::bitstream::{BitStackReader, BitStackWriter};
use crate::histogram::NormHistogram;
use crate::{find_mask, TABLE_LOG_RANGE};

// Stepping through by 5/8 * size + 3 will, thanks to the "3" part,
// uniquely step through the entire table. There are marginally better
// ways to distribute these symbols, but this one's very fast.
#[inline]
fn table_step(size: usize) -> usize {
    size * 5 / 8 + 3
}

#[derive(Clone, Debug)]
pub struct EncodeTable {
    table_log: u32,
    table: Vec<u16>,
    symbol_tt: [SymbolTransform; 256],
    symbols: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
struct SymbolTransform {
    bits: u32,
    find_state: i32,
}

impl EncodeTable {
    /// Construct a new FSE table from a provided normalized histogram.
    pub fn new(hist: &NormHistogram) -> Self {
        let size = 1 << hist.log2_sum();
        let mut this = Self {
            table_log: hist.log2_sum(),
            table: Vec::with_capacity(size),
            symbol_tt: [SymbolTransform::default(); 256],
            symbols: Vec::with_capacity(size),
        };
        this.update(hist);
        this
    }

    /// Replace the current FSE table with one from a new normalized, histogram, reusing internal buffers as needed.
    pub fn update(&mut self, hist: &NormHistogram) {
        // Shouldn't be possible, as the histogram normalization also does this check.
        assert!(
            TABLE_LOG_RANGE.contains(&hist.log2_sum()),
            "FSE Table must be between 2^9 to 2^16"
        );
        self.table_log = hist.log2_sum();
        let size = 1 << self.table_log;
        let mut cumul = [0u32; 256];
        let mut high_threshold = size - 1;
        self.symbols.clear();
        self.symbols.resize(size, 0);

        // First, we build up the start positions of each symbol in the "cumul"
        // array. We special-case handle low-probability values by pre-filling
        // the symbol table from the highest spot down with that symbol's index.
        // This means weighting the low-prob symbols towards the end of the
        // state table.
        let mut acc = 0;
        for (i, (&x, c)) in hist.table_iter().zip(cumul.iter_mut()).enumerate() {
            *c = acc;
            if x == -1 {
                acc += 1;
                self.symbols[high_threshold] = i as u8;
                high_threshold -= 1;
            } else {
                acc += x as u32;
            }
        }

        // Next, we spread the symbols across the symbol table.
        //
        // We start at position 0. Then, for each count in the normalized
        // histogram, we fill in the symbol value at our current position, then
        // increment the position pointer by our step. When the position pointer
        // is within the table but inside the "low probability area", we
        // increment until we're outside of it.
        //
        let mut position = 0;
        let table_mask = size - 1;
        let step = table_step(size);
        for (i, &x) in hist.table_iter().enumerate() {
            for _ in 0..x {
                self.symbols[position] = i as u8;
                position = (position + step) & table_mask;
                while position > high_threshold {
                    position = (position + step) & table_mask;
                }
            }
        }
        assert!(position == 0); // We should have completely stepped through the entire table.

        // After we spread the symbols, it's time to build the actual encoding tables.
        // For each point in the table, we look up what the symbol value at that
        // spot is. We then write in the next state value, and increment the
        // table offset for that particular symbol.
        self.table.resize(size, 0);
        for (i, &x) in self.symbols.iter().enumerate() {
            let x = x as usize;
            self.table[cumul[x] as usize] = (size + i) as u16;
            cumul[x] += 1;
        }

        // Build Symbol Transformation Table
        self.symbol_tt.fill(SymbolTransform::default());
        let mut total = 0;
        for (&x, tt) in hist.table_iter().zip(self.symbol_tt.iter_mut()) {
            match x {
                // We fill this anyway for potential future use with a max symbol cost estimator
                0 => tt.bits = ((self.table_log + 1) << 16) - (1 << self.table_log),
                -1 | 1 => {
                    *tt = SymbolTransform {
                        bits: (self.table_log << 16) - (1 << self.table_log),
                        find_state: total - 1,
                    };
                    total += 1;
                }
                x => {
                    let max_bits_out = self.table_log - (x - 1).ilog2();
                    let min_state_plus = (x as u32) << max_bits_out;
                    *tt = SymbolTransform {
                        bits: (max_bits_out << 16) - min_state_plus,
                        find_state: total - x,
                    };
                    total += x;
                }
            }
        }
    }

    pub fn compress_bound(size: usize) -> usize {
        512 + size + (size >> 7) + 4 + std::mem::size_of::<usize>()
    }
}

#[derive(Debug)]
pub struct Encoder<'a> {
    value: u32,
    table: &'a EncodeTable,
}

impl<'a> Encoder<'a> {
    pub fn new(table: &'a EncodeTable) -> Self {
        Self { value: 0, table }
    }

    /// Same as [`Encoder::new`], except that the first symbol to encode (and
    /// thus the last that will be read) uses the smallest state value possible,
    /// saving the cost of this symbol.
    pub fn new_first_symbol(table: &'a EncodeTable, first_symbol: u8) -> Self {
        let mut this = Self::new(table);
        let symbol_tt = this.table.symbol_tt[first_symbol as usize];
        let bits_out = (symbol_tt.bits + (1 << 15)) >> 16;
        this.value = (bits_out << 16) - symbol_tt.bits;
        let idx = ((this.value >> bits_out) as i32 + symbol_tt.find_state) as usize;
        this.value = this.table.table[idx] as u32;
        this
    }

    /// Encode a symbol, without ensuring the writer has space to accept the new
    /// bits.
    ///
    /// # Safety
    /// This function doesn't ensure the writer has space to accept the new
    /// symbol bits.
    ///
    pub unsafe fn encode_raw(&mut self, writer: &mut BitStackWriter<'_>, sym: u8) {
        let symbol_tt = self.table.symbol_tt[sym as usize];
        let bits_out = (symbol_tt.bits + self.value) >> 16;
        writer.write_bits_raw_unmasked(self.value as usize, bits_out as usize);
        let idx = ((self.value >> bits_out) as i32 + symbol_tt.find_state) as usize;
        //println!(
        //    "write byte {:02x}, using {} bits, put 0x{:04x} on the stream",
        //    sym,
        //    bits_out,
        //    self.value as usize & find_mask(bits_out as usize)
        //);
        self.value = *self.table.table.get_unchecked(idx) as u32;
    }

    /// Encode a symbol
    pub fn encode(&mut self, writer: &mut BitStackWriter<'_>, sym: u8) {
        writer.flush();
        unsafe { self.encode_raw(writer, sym) }
    }

    /// Complete the encoding by appending the final state value of the encoder
    pub fn finish(self, writer: &mut BitStackWriter<'_>) {
        writer.write_bits_unmasked(self.value as usize, self.table.table_log as usize);
    }
}

#[derive(Clone, Debug)]
pub struct DecodeTable {
    table_log: u32,
    fast_mode: bool,
    table: Vec<DecodeTransform>,
}

#[derive(Clone, Copy, Debug, Default)]
struct DecodeTransform {
    new_state: u16,
    symbol: u8,
    num_bits: u8,
}

impl DecodeTable {
    /// Create a new FSE decoding table from a normalized histogram.
    pub fn new(hist: &NormHistogram) -> Self {
        let mut this = Self {
            table_log: 0,
            fast_mode: false,
            table: Vec::with_capacity(1 << hist.log2_sum()),
        };
        this.update(hist);
        this
    }

    /// Update this FSE decoding table for a new normalized histogram.
    pub fn update(&mut self, hist: &NormHistogram) {
        // Shouldn't be possible, as the histogram normalization also does this check.
        assert!(
            TABLE_LOG_RANGE.contains(&hist.log2_sum()),
            "FSE Table must be between 2^9 to 2^16"
        );

        // Prepare to load in the new table
        self.table_log = hist.log2_sum();
        let size = 1 << self.table_log;
        self.fast_mode = true;
        self.table.clear();
        self.table.resize(size, DecodeTransform::default());

        // Load in the low-probability symbols
        let mut symbol_next = [0u16; 256];
        let large_limit = 1u16 << (self.table_log - 1);
        let mut high_threshold = size - 1;
        for (s, (&c, sym)) in hist.table_iter().zip(symbol_next.iter_mut()).enumerate() {
            if c <= -1 {
                self.table[high_threshold].symbol = s as u8;
                high_threshold -= 1;
                *sym = 1;
            } else {
                let c = c as u16; // Range is now 0-32768, so this should be fine.
                if c >= large_limit {
                    self.fast_mode = false;
                }
                *sym = c;
            }
        }

        // Spread the symbols across the table, using the same mechanism as the compressor did
        let mut position = 0;
        let table_mask = size - 1;
        let step = table_step(size);
        for (s, &c) in hist.table_iter().enumerate() {
            for _ in 0..c {
                self.table[position].symbol = s as u8;
                position = (position + step) & table_mask;
                while position > high_threshold {
                    position = (position + step) & table_mask;
                }
            }
        }

        assert!(position == 0, "Table wasn't completely initialized somehow");

        // Build the decoding table
        for decode in self.table.iter_mut() {
            let sym = decode.symbol;
            let next_state = symbol_next[sym as usize];
            symbol_next[sym as usize] += 1;
            let num_bits = (self.table_log - next_state.ilog2()) as u8;
            let new_state = (((next_state as u32) << num_bits) - (size as u32)) as u16;
            decode.num_bits = num_bits;
            decode.new_state = new_state;
        }
    }
}

#[derive(Debug)]
pub struct Decoder<'a> {
    state: u16,
    table: &'a DecodeTable,
}

impl<'a> Decoder<'a> {
    /// Initialize a stream decoder, failing if there aren't enough bits in the reader.
    pub fn new(table: &'a DecodeTable, reader: &mut BitStackReader) -> Option<Self> {
        let state = reader.read(table.table_log as usize)? as u16;
        Some(Self { state, table })
    }

    fn state_lookup(&self) -> &DecodeTransform {
        unsafe { self.table.table.get_unchecked(self.state as usize) }
    }

    /// Decode the next symbol without reloading the reader's internal buffer
    ///
    /// # Safety
    /// Only do this if you're certain there will be enough bits left in the
    /// internal buffer for future operations.
    pub unsafe fn decode_symbol_no_reload(&mut self, reader: &mut BitStackReader) -> Option<u8> {
        let state_info = self.state_lookup();
        let low_bits = reader.read_no_reload(state_info.num_bits as usize)?;
        let sym = state_info.symbol;
        //println!(
        //    "Read byte {:02x}, take {} bits, got 0x{:04x} off the stream",
        //    sym, state_info.num_bits, low_bits
        //);
        self.state = state_info.new_state + (low_bits as u16);
        Some(sym)
    }

    /// Decode the next symbol and pull more data from the reader.
    pub fn decode_symbol(&mut self, reader: &mut BitStackReader) -> Option<u8> {
        let sym = unsafe { self.decode_symbol_no_reload(reader)? };
        reader.reload();
        Some(sym)
    }

    /// Extract the final symbol from the current state and finish decoding.
    pub fn finish(self) -> u8 {
        self.state_lookup().symbol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Run the compressor directly
    pub fn fse_compress(src: &[u8], dst: &mut Vec<u8>) -> (NormHistogram, usize) {
        let mut writer = BitStackWriter::new(dst);
        let hist = NormHistogram::new(src);
        let fse_table = EncodeTable::new(&hist);
        let mut src_iter = src.chunks(2).rev();
        let first = src_iter.next().unwrap();
        let first_byte = first.last().unwrap();
        let mut encode = Encoder::new_first_symbol(&fse_table, *first_byte);
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

    // Run the decompressor directly
    pub fn fse_decompress(hist: &NormHistogram, src: &[u8], dst: &mut Vec<u8>) -> Option<usize> {
        let len = dst.len();
        let mut reader = BitStackReader::new(src)?;
        let fse_table = DecodeTable::new(hist);
        let mut decode = Decoder::new(&fse_table, &mut reader).unwrap();
        while let Some(s) = decode.decode_symbol(&mut reader) {
            dst.push(s);
        }
        dst.push(decode.finish());
        Some(dst.len() - len)
    }

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

    fn compact_byte_slice_string(slice: &[u8]) -> String {
        use std::fmt::Write;
        let mut s = String::with_capacity(2 + 3 * slice.len());
        s.push('[');
        for b in slice {
            write!(s, "{:02x},", b).unwrap();
        }
        s.push(']');
        s
    }

    #[test]
    fn decompress() {
        let src = gen_sequence(0.2, 1 << 15);
        let mut dst = Vec::with_capacity(src.len() + 8);
        let mut dec = Vec::with_capacity(src.len());
        let (hist, _) = fse_compress(&src, &mut dst);
        fse_decompress(&hist, &dst, &mut dec).expect("Failed to decompress");
        if src != dec {
            println!("src.len = {}, dec.len = {}", src.len(), dec.len());
            println!(
                "Start of src: {}",
                compact_byte_slice_string(&src[..16.min(src.len())])
            );
            println!(
                "Start of dec: {}",
                compact_byte_slice_string(&dec[..16.min(dec.len())])
            );
            println!(
                "End of src: {}",
                compact_byte_slice_string(&src[src.len().saturating_sub(17)..])
            );
            println!(
                "End of dec: {}",
                compact_byte_slice_string(&dec[dec.len().saturating_sub(17)..])
            );
            panic!("Decoded data doesn't match up with encoded data")
        }
    }
}
