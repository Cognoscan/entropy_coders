use histogram::NormHistogram;

pub mod bitstream;
pub mod histogram;

const TABLE_LOG_MIN: u32 = 5;
const TABLE_LOG_MAX: u32 = 15;
const TABLE_LOG_RANGE: std::ops::RangeInclusive<u32> = TABLE_LOG_MIN..=TABLE_LOG_MAX;

pub struct FseTable {
    table_log: u32,
    table: Vec<u16>,
    symbol_tt: [SymbolTransform; 256],
}

#[derive(Clone, Copy, Debug)]
struct SymbolTransform {
    bits: u32,
    find_state: u32,
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
                0 => tt.bits = ((table_log+1)<<16) - (1<<table_log),
                -1 | 1 => {
                    *tt = SymbolTransform {
                        bits: (table_log<<16) - (1<<table_log),
                        find_state: total - 1,
                    };
                    total += 1;
                }
                x => {
                    let x = x as u32;
                    let max_bits_out = table_log - (31 - (x-1).leading_zeros());
                    let min_state_plus = x << max_bits_out;
                    *tt = SymbolTransform {
                        bits: (max_bits_out << 1) - min_state_plus,
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
}
