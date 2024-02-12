use core::num;

use crate::bitstream::{BitStackReader, BitStackWriter};
use crate::histogram::NormHistogram;
use crate::TABLE_LOG_RANGE;

// Stepping through by 5/8 * size + 3 will, thanks to the "3" part,
// uniquely step through the entire table. There are marginally better
// ways to distribute these symbols, but this one's very fast.
#[inline]
fn table_step(size: usize) -> usize {
    size * 5 / 8 + 3
}

#[derive(Clone, Debug)]
pub struct FseEncodeTable {
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

impl FseEncodeTable {

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
        for (i, &x) in hist
            .table_iter()
            .enumerate()
        {
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
        for (&x, tt) in hist
            .table_iter()
            .zip(self.symbol_tt.iter_mut())
        {
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
pub struct FseEncode<'a> {
    value: u32,
    table: &'a FseEncodeTable,
}

impl<'a> FseEncode<'a> {
    pub fn new(table: &'a FseEncodeTable) -> Self {
        Self { value: 0, table }
    }

    /// Same as [`new`], except that the first symbol to encode (and thus the
    /// last that will be read) uses the smallest state value possible, saving
    /// the cost of this symbol.
    pub fn new_first_symbol(table: &'a FseEncodeTable, first_symbol: u8) -> Self {
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

#[derive(Clone, Debug)]
pub struct FseDecodeTable {
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

impl FseDecodeTable {

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
        let large_limit = 1u16 << (self.table_log-1);
        let mut high_threshold = size-1;
        for (s, (&c, sym)) in hist.table_iter().zip(symbol_next.iter_mut()).enumerate() {
            if c <= -1 {
                self.table[high_threshold].symbol = s as u8;
                high_threshold -= 1;
                *sym = 1;
            }
            else {
                let c = c as u16; // Range is now 0-32768, so this should be fine.
                if c >= large_limit { self.fast_mode = false; }
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
                while position > high_threshold { position = (position + step) & table_mask; }
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
pub struct FseDecode<'a> {
    value: u32,
    table: &'a FseEncodeTable,
}

impl<'a> FseDecode<'a> {

    /// Initialize a stream decoder
    pub fn new(table: &'a FseEncodeTable, reader: &mut BitStackReader) -> std::io::Result<Self> {
        let value = reader.read(table.table_log as usize)? as u32;
        Ok(Self { value, table })
    }
}