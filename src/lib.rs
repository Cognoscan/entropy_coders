use bitstream::BitStackWriter;
use histogram::NormHistogram;

pub mod bitstream;
pub mod histogram;

const TABLE_LOG_MIN: u32 = 5;
const TABLE_LOG_MAX: u32 = 15;
const TABLE_LOG_RANGE: std::ops::RangeInclusive<u32> = TABLE_LOG_MIN..=TABLE_LOG_MAX;
const TABLE_LOG_DEFAULT: u32 = 11;

#[derive(Clone, Debug)]
pub struct FseTable {
    table_log: u32,
    table: Vec<u16>,
    symbol_tt: [SymbolTransform; 256],
}

#[derive(Clone, Copy, Debug)]
struct SymbolTransform {
    bits: u32,
    find_state: i32,
}

impl FseTable {
    pub fn new(hist: &NormHistogram) -> Self {
        assert!(
            TABLE_LOG_RANGE.contains(&hist.log2_sum()),
            "FSE Table must be between 2^9 to 2^16"
        );

        let table_log = hist.log2_sum();
        let size = 1 << table_log;
        let mut cumul = [0u32; 256];
        let mut symbols = vec![0; size];
        let mut high_threshold = size - 1;

        // First, we build up the start positions of each symbol in the "cumul"
        // array. We special-case handle low-probability values by pre-filling
        // the symbol table from the highest spot down with that symbol's index.
        // This means weighting the low-prob symbols towards the end of the
        // state table.
        let mut acc = 0;
        for (i, (&x, c)) in hist.table().iter().zip(cumul.iter_mut()).enumerate() {
            *c = acc;
            if x == -1 {
                acc += 1;
                symbols[high_threshold] = i;
                high_threshold -= 1;
            } else {
                acc += x as u32;
            }
        }

        // Next, we spread the symbols across the symbol table.
        //
        // Our step is "5/8 * size + 3", which this does with a pair of shifts and adds. Ok.
        // We start at position 0. Then, for each count in the normalized
        // histogram, we fill in the symbol value at our current position, then
        // increment the position pointer by our step. When the position pointer
        // is within the table but inside the "low probability area", we
        // increment until we're outside of it.
        //
        // Stepping through by 5/8 * size + 3 will, thanks to the "3" part,
        // uniquely step through the entire table. There are marginally better
        // ways to distribute these symbols, but this one's very fast.
        let mut position = 0;
        let table_mask = size - 1;
        let step = size * 5 / 8 + 3;
        for (i, &x) in hist
            .table()
            .iter()
            .take(hist.max_symbol() as usize + 1)
            .enumerate()
        {
            for _ in 0..x {
                symbols[position] = i;
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
        let mut table = vec![0u16; size];
        for (i, &x) in symbols.iter().enumerate() {
            table[cumul[x] as usize] = (size + i) as u16;
            cumul[x] += 1;
        }

        // Build Symbol Transformation Table
        let mut symbol_tt = [SymbolTransform {
            bits: 0,
            find_state: 0,
        }; 256];
        let mut total = 0;
        for (&x, tt) in hist
            .table()
            .iter()
            .zip(symbol_tt.iter_mut())
            .take(hist.max_symbol() as usize + 1)
        {
            match x {
                // We fill this anyway for potential future use with a max symbol cost estimator
                0 => tt.bits = ((table_log + 1) << 16) - (1 << table_log),
                -1 | 1 => {
                    *tt = SymbolTransform {
                        bits: (table_log << 16) - (1 << table_log),
                        find_state: total - 1,
                    };
                    total += 1;
                }
                x => {
                    let max_bits_out = table_log - (x - 1).ilog2();
                    let min_state_plus = (x as u32) << max_bits_out;
                    *tt = SymbolTransform {
                        bits: (max_bits_out << 16) - min_state_plus,
                        find_state: total - x,
                    };
                    total += x;
                }
            }
        }

        FseTable {
            table_log,
            table,
            symbol_tt,
        }
    }

    pub fn compress_bound(size: usize) -> usize {
        512 + size + (size >> 7) + 4 + std::mem::size_of::<usize>()
    }
}

#[derive(Debug)]
pub struct FseEncode<'a> {
    value: u32,
    table: &'a FseTable,
}

impl<'a> FseEncode<'a> {
    pub fn new(table: &'a FseTable) -> Self {
        Self { value: 0, table }
    }

    /// Same as [`new`], except that the first symbol to encode (and thus the
    /// last that will be read) uses the smallest state value possible, saving
    /// the cost of this symbol.
    pub fn new_first_symbol(table: &'a FseTable, first_symbol: u8) -> Self {
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

/// Run the compressor directly
pub fn fse_compress(src: &[u8], dst: &mut Vec<u8>) -> usize {
    let mut writer = BitStackWriter::new(dst);
    let hist = histogram::NormHistogram::new(src);
    let fse_table = FseTable::new(&hist);
    let mut src_iter = src.iter().rev();
    let first = src_iter.next().unwrap();
    let mut encode = FseEncode::new_first_symbol(&fse_table, *first);
    for n in src_iter {
        unsafe {
            encode.encode_raw(&mut writer, *n);
            if usize::BITS < 64 { writer.flush(); }
            encode.encode_raw(&mut writer, *n);
        }
        writer.flush();
    }
    encode.finish(&mut writer);
    writer.finish()
}

/// Compress with two streams at once
pub fn fse_compress2(src: &[u8], dst: &mut Vec<u8>) -> usize {
    let mut writer = BitStackWriter::new(dst);
    let hist = histogram::NormHistogram::new(src);
    let fse_table = FseTable::new(&hist);
    let mut src_iter = src.chunks(4).rev();
    let first = src_iter.next().unwrap();
    let (mut encode0, mut encode1) = match first.len() {
        1 => {
            let next = src_iter.next().unwrap();
            let mut encode0 = FseEncode::new_first_symbol(&fse_table, first[0]);
            let mut encode1 = FseEncode::new_first_symbol(&fse_table, next[3]);
            encode0.encode(&mut writer, next[2]);
            encode1.encode(&mut writer, next[1]);
            encode0.encode(&mut writer, next[0]);
            (encode0, encode1)
        },
        2 => {
            let encode1 = FseEncode::new_first_symbol(&fse_table, first[1]);
            let encode0 = FseEncode::new_first_symbol(&fse_table, first[0]);
            (encode0, encode1)
        },
        3 => {
            let mut encode0 = FseEncode::new_first_symbol(&fse_table, first[2]);
            let encode1 = FseEncode::new_first_symbol(&fse_table, first[1]);
            encode0.encode(&mut writer, first[0]);
            (encode0, encode1)
        },
        4 => {
            let mut encode1 = FseEncode::new_first_symbol(&fse_table, first[3]);
            let mut encode0 = FseEncode::new_first_symbol(&fse_table, first[2]);
            encode1.encode(&mut writer, first[1]);
            encode0.encode(&mut writer, first[0]);
            (encode0, encode1)
        },
        _ => panic!("Should always be a chunk of 4"),
    };

    for n in src_iter {
        unsafe {
          encode1.encode_raw(&mut writer, *n.get_unchecked(3));
          encode0.encode_raw(&mut writer, *n.get_unchecked(2));
          encode1.encode_raw(&mut writer, *n.get_unchecked(1));
          encode0.encode_raw(&mut writer, *n.get_unchecked(0));
          writer.flush();
        }
    }

    encode0.finish(&mut writer);
    encode1.finish(&mut writer);
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
