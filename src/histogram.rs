use std::io::Write;
use thiserror::Error;

use crate::{
    bitstream::{BitStackWriter, BitStreamReader},
    TABLE_LOG_MAX, TABLE_LOG_MIN, TABLE_LOG_RANGE,
};

/// A histogram of all the 8-bit symbols seen in a data sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram {
    table: [u32; 256],
    size: u32,
    max_symbol: u8,
}

impl Histogram {
    /// Create a new histogram. Panics if data is >= 4 GiB.
    pub fn new(data: &[u8]) -> Self {
        assert!(data.len() <= u32::MAX as usize, "Data vector is too long");
        const P: usize = 4;
        let mut tables = [[0u32; 256]; P];
        let mut iter = data.chunks_exact(P);
        for i in &mut iter {
            for j in 0..P {
                // SAFETY: i[j] is a u8 and thus 0-255, and our table is 256 long. It
                // will *never* be out of range. j is guaranteed to be in range
                // thanks to chunks_exact.
                unsafe {
                    *(tables
                        .get_unchecked_mut(j)
                        .get_unchecked_mut(*i.get_unchecked(j) as usize)) += 1;
                }
            }
        }

        let rem = iter.remainder();
        for j in 0..P {
            if let Some(v) = rem.get(j) {
                unsafe {
                    *(tables.get_unchecked_mut(j).get_unchecked_mut(*v as usize)) += 1;
                }
            }
        }

        let mut table = [0u32; 256];
        for (i, v) in table.iter_mut().enumerate() {
            for t in tables.iter() {
                *v += t[i];
            }
        }

        let mut max_symbol = 0;
        for (i, x) in table.iter().enumerate().rev() {
            if *x != 0 {
                max_symbol = i as u8;
                break;
            }
        }

        Self {
            table,
            size: data.len() as u32,
            max_symbol,
        }
    }

    /// Iterate over the populated part of the symbol frequency table
    pub fn table_iter(&self) -> impl Iterator<Item = &u32> {
        self.table.iter().take(self.max_symbol as usize + 1)
    }

    /// Get the symbol frequency table
    pub fn table(&self) -> &[u32; 256] {
        &self.table
    }

    /// How many unique symbols are in the table
    pub fn symbol_count(&self) -> usize {
        self.table.iter().map(|x| if *x == 0 { 1 } else { 0 }).sum()
    }

    /// What the highest-numbered symbol seen was
    pub fn max_symbol(&self) -> u8 {
        self.max_symbol
    }

    /// Get the number of symbols that were in the original data
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Normalize the histogram. Panics if log2 is not between 5 and 15.
    pub fn normalize(self, log2: u32) -> NormHistogram {
        assert!(
            TABLE_LOG_RANGE.contains(&log2),
            "Histogram normalization: log2 must be between 5 & 15, but is {}",
            log2
        );

        const RTB_TABLE: [u32; 8] = [0, 473195, 504333, 520860, 550000, 700000, 750000, 830000];

        let scale = 62u64 - (log2 as u64);
        let step = (1u64 << 62) / (self.size as u64);
        let v_step = 1u64 << (scale - 20);
        let low_threshold = self.size >> log2;
        let mut to_distribute = 1 << (log2 as i32);
        let mut largest = 0;
        let mut largest_prob = 0;

        let mut table = [0i32; 256];

        for (i, (&t, t_norm)) in self.table_iter().zip(table.iter_mut()).enumerate() {
            if t == self.size {
                *t_norm = to_distribute;
                return NormHistogram {
                    table,
                    log2,
                    max_symbol: self.max_symbol,
                };
            }
            if t == 0 {
                continue;
            }
            if t <= low_threshold {
                *t_norm = -1;
                to_distribute -= 1;
                continue;
            }
            let mut prob = (t as u64 * step) >> scale;
            if prob < 8 {
                let rest_to_beat = v_step * (RTB_TABLE[prob as usize] as u64);
                prob += ((t as u64 * step - (prob << scale)) > rest_to_beat) as u64;
            }
            let prob = prob as i32;
            if prob > largest_prob {
                largest_prob = prob;
                largest = i;
            }
            *t_norm = prob;
            to_distribute -= prob;
        }

        // Check if we're in an unusual probability distribution
        if to_distribute != 0 && -to_distribute >= (largest_prob >> 1) {
            return self.normalize_slow(log2);
        } else {
            table[largest] += to_distribute;
        }

        NormHistogram {
            table,
            log2,
            max_symbol: self.max_symbol,
        }
    }

    fn normalize_slow(self, log2: u32) -> NormHistogram {
        println!("normalizing slowly");
        const UNASSIGNED: i32 = -2;
        let low_threshold = self.size >> log2;
        let low_one = (self.size * 3) >> (log2 + 1);
        let mut table = [0i32; 256];
        let mut to_distribute = 1 << (log2 as i32);
        let mut total = self.size;

        // Begin by distributing the low-probability symbols
        for (&t, t_norm) in self.table_iter().zip(table.iter_mut()) {
            if t == 0 {
                continue;
            } else if t <= low_threshold {
                *t_norm = -1;
                to_distribute -= 1;
                total -= t;
            } else if t <= low_one {
                *t_norm = 1;
                to_distribute -= 1;
                total -= t;
            } else {
                *t_norm = UNASSIGNED;
            }
        }

        if to_distribute == 0 {
            return NormHistogram {
                table,
                log2,
                max_symbol: self.max_symbol,
            };
        }

        // Do another round of distributing low-probability values
        if (total / to_distribute) > low_one {
            let low = (total * 3) / (to_distribute * 2);
            for (&t, t_norm) in self
                .table
                .iter()
                .zip(table.iter_mut())
                .take(self.max_symbol as usize + 1)
            {
                if *t_norm == UNASSIGNED && t <= low {
                    *t_norm = 1;
                    to_distribute -= 1;
                    total -= t
                }
            }
        }

        if ((1 << log2) - to_distribute) == (self.max_symbol as u32 + 1) {
            // We marked every single probability evenly, which means this is
            // functionally incompressible data. Just find the max, hand out the
            // remaining points to it, and call it a day.
            let mut v_max = 0;
            let mut i_max = 0;
            for (i, &v) in self.table.iter().enumerate() {
                if v > v_max {
                    v_max = v;
                    i_max = i;
                }
            }
            table[i_max] += to_distribute as i32;
            return NormHistogram {
                table,
                log2,
                max_symbol: self.max_symbol,
            };
        } else if total == 0 {
            // If all values are pretty poor, just evenly spread the values across
            // the remainder
            while to_distribute != 0 {
                // Repeat until we're out of numbers to distribute.
                for t in table.iter_mut().take(self.max_symbol as usize + 1) {
                    if *t > 0 {
                        *t += 1;
                        to_distribute -= 1;
                        if to_distribute == 0 {
                            break;
                        }
                    }
                }
            }
        } else {
            let v_step_log: u64 = 62 - (log2 as u64);
            let mid: u64 = (1 << (v_step_log - 1)) - 1;
            let r_step: u64 = (((1 << v_step_log) * (to_distribute as u64)) + mid) / (total as u64);
            let mut tmp_total = mid;
            for (&t, t_norm) in self
                .table
                .iter()
                .zip(table.iter_mut())
                .take(self.max_symbol as usize + 1)
            {
                if *t_norm == UNASSIGNED {
                    let end = tmp_total + (t as u64 * r_step);
                    let s_start = tmp_total >> v_step_log;
                    let s_end = end >> v_step_log;
                    let weight = s_end - s_start;
                    if weight < 1 {
                        panic!("What did you do, to make a distribution so cursed");
                    }
                    *t_norm = weight as i32;
                    tmp_total = end;
                }
            }
        }

        NormHistogram {
            table,
            log2,
            max_symbol: self.max_symbol,
        }
    }
}

/// A normalized histogram. Uses the special value "-1" to indicate a
/// probability that was less than 1 in the original histogram.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormHistogram {
    table: [i32; 256],
    log2: u32,
    max_symbol: u8,
}

impl NormHistogram {
    // Get the normalized table
    pub fn table(&self) -> &[i32; 256] {
        &self.table
    }

    /// Iterate over the populated part of the symbol frequency table
    pub fn table_iter(&self) -> impl Iterator<Item = &i32> {
        self.table.iter().take(self.max_symbol as usize + 1)
    }

    // Get the base-2 logarithm of the sum of the table.
    pub fn log2_sum(&self) -> u32 {
        self.log2
    }

    /// How many unique symbols are in the table
    pub fn symbol_count(&self) -> usize {
        self.table.iter().map(|x| if *x == 0 { 1 } else { 0 }).sum()
    }

    /// What the highest-numbered symbol seen was
    pub fn max_symbol(&self) -> u8 {
        self.max_symbol
    }

    pub fn write_bound(&self) -> usize {
        let max_header_size = (((self.max_symbol as usize + 1) * (self.log2 as usize)) >> 3) + 3;
        if self.max_symbol > 0 {
            max_header_size
        } else {
            512
        }
    }

    /// Append the normalized histogram to a byte vector.
    ///
    /// The write format follows that of `zstd`, repeated here for convenience.
    /// First, the log2(size) value used to make the table is written as
    /// `log2_size-5`, as a 4 bit number.  Next is the sequence of symbol
    /// values, from symbol 0 to the last present one. The number of bits used
    /// by each symbol value is variable, and depends on the running count of
    /// all values encoded so far. Each value encoded is 1 plus the symbol count;
    /// this is because "low-probability" symbols are stored as "-1" and thus we
    /// need to offset the value by 1 to get a non-negative value for encoding.
    /// When the sum of all symbol counts is equal to `size`, the table is done.
    ///
    /// When a count of zero (value of 1) is encoded, the next 2 bits are a
    /// "repeat" marker, indicating that the next 0-3 symbols also have a count
    /// of zero. If the repeat marker is set to 3, the next 2 bits are also a
    /// repeat marker, and so on.
    ///
    /// Because we can only encode some number of bits, but the available range
    /// for N bits is always a power of two, we use the "spare" space to encode
    /// some values with 1 bit less than might otherwise be needed. Let
    /// `threshold=ceil(log2(remaining_counts))` be true. Then the spare space
    /// is `threshold-remaining_counts`, and that can be used to encode 0 to
    /// `threshold-remaining_counts` with one less bit than the other possible.
    /// Ripping from the `zstd` documentation, take an example where there are
    /// 156 values left. That means values from 0-157 (inclusive) are possible,
    /// and thus 255-157=98 values are free to use in an 8-bit field. The first
    /// 98 values (0-97) use 7 bits, and values from 98-157 use 8 bits. This
    /// creates the encoding scheme:
    ///
    /// | Value read | Value decoded | Number of bits used |
    /// | --         | --            | --                  |
    /// | 0 - 97     | 0 - 97        | 7                   |
    /// | 98 - 127   | 98 - 127      | 8                   |
    /// | 128 - 225  | 0 - 97        | 7                   |
    /// | 226 - 255  | 128 - 157     | 8                   |
    ///
    pub fn write<T: Write>(&self, writer: &mut T) -> std::io::Result<usize> {
        let mut writer = BitStackWriter::new(writer);

        // Write out the table's log2 size
        let write_size = self.log2 - TABLE_LOG_MIN;
        writer.write_bits(write_size as usize, 4)?;

        let mut threshold = 1 << self.log2;
        let mut remaining = threshold + 1;
        let mut zero_count = 0;
        let mut num_bits = (self.log2 + 1) as usize;
        for &s in self.table_iter() {
            if remaining <= 1 {
                break;
            }
            if zero_count != 0 {
                if s == 0 {
                    zero_count += 1;
                    continue;
                }

                // Write out some number of 2-bit "repeat" markers, used for
                // indicating a run of zeros.
                zero_count -= 1;
                while zero_count >= 24 {
                    writer.write_bits(0xFFFF, 16)?;
                    zero_count -= 24;
                }
                while zero_count >= 3 {
                    writer.write_bits(0x3, 2)?;
                    zero_count -= 3;
                }
                writer.write_bits(zero_count, 2)?;
            }
            let max = (2 * threshold - 1) - remaining;
            remaining -= s.abs(); // Subtract out the count from our remaining count
            let mut count = s + 1;
            if count >= threshold {
                count += max;
            }
            let bits_to_write = num_bits - (count < max) as usize;
            writer.write_bits(count as usize, bits_to_write)?;
            zero_count = (count == 1) as usize;
            if remaining < 1 {
                panic!("Normalized histogram was incorrect somehow");
            }
            // Adjust downward to the minimum number of bits needed for encoding
            // a remaining value.
            while remaining < threshold {
                num_bits -= 1;
                threshold >>= 1;
            }
        }

        writer.finish()
    }

    /// Read in a normalized histogram that has been written out with [`write`][Self::write].
    pub fn read(slice: &[u8]) -> Result<(Self, &[u8]), HistError> {
        let mut reader = BitStreamReader::new(slice, slice.len() * 8);
        let log2 = reader.read(4)? as u32 + TABLE_LOG_MIN;
        if log2 > TABLE_LOG_MAX {
            return Err(HistError::TableLogTooLarge(log2));
        }
        let mut hist = Self {
            log2,
            table: [0; 256],
            max_symbol: 255,
        };
        let mut symbol = 0;
        let mut threshold = 1 << hist.log2;
        let mut remaining = threshold + 1;
        let mut read_bit_count = log2 as usize + 1;
        let mut previous0 = false;

        while remaining > 1 && symbol < 256 {
            // Read in the 2-bit zero-continuation marks
            if previous0 {
                while reader.peek(16).unwrap_or(0) == 0xFFFF {
                    reader.advance_by(16)?;
                    symbol += 24;
                }
                while reader.peek(2).unwrap_or(0) == 3 {
                    reader.advance_by(2)?;
                    symbol += 3;
                }
                symbol += reader.read(2)?;
            }
            if symbol >= 256 {
                break;
            }

            let max = (2 * threshold - 1) - remaining;
            let raw_value = reader
                .peek(read_bit_count)
                .or_else(|_| reader.peek(read_bit_count - 1))?;
            let mut value;
            if (raw_value & (threshold - 1)) < max {
                reader.advance_by(read_bit_count - 1)?;
                value = raw_value & (threshold - 1);
            } else {
                reader.advance_by(read_bit_count)?;
                value = raw_value & (2 * threshold - 1);
                if value >= threshold {
                    value -= max
                };
            }
            let value = value as i32 - 1;
            // Subtract out the retrieved count. Note: this will not ever
            // overflow, as our decoding method ensures `value` is never more than `remaining-1`.
            remaining -= value.unsigned_abs() as usize;
            hist.table[symbol] = value;
            symbol += 1;
            previous0 = value == 0;
            while remaining < threshold {
                read_bit_count -= 1;
                threshold >>= 1;
            }
        }

        if remaining != 1 {
            return Err(HistError::TooManySymbols);
        }

        hist.max_symbol = (symbol - 1) as u8;

        Ok((hist, reader.finish_byte()))
    }
}

#[derive(Debug, Error)]
pub enum HistError {
    #[error("Table log2 size is {0}, higher than the accepted maximum")]
    TableLogTooLarge(u32),
    #[error("Histogram counts are spread across more than 256 symbols")]
    TooManySymbols,
    #[error("Read error")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Fill;

    fn hist_verify(hist: Histogram, log2: u32) {
        print!("hist = [ ");
        for &x in hist.table() {
            print!("{}, ", x);
        }
        println!("]");
        let hist_table = hist.table().to_owned();
        let norm_hist = hist.normalize(log2);
        print!("norm_hist = [ ");
        for &x in norm_hist.table() {
            print!("{}, ", x);
        }
        println!("]");
        let sum: i32 = norm_hist.table.iter().map(|x| x.abs()).sum();
        assert_eq!(sum, 1 << log2);
        for (i, (h, hn)) in hist_table.iter().zip(norm_hist.table().iter()).enumerate() {
            assert_eq!(
                *h == 0,
                *hn == 0,
                "symbol {} histogram count was {} but normalized count is {}",
                i,
                h,
                hn
            );
        }

        let mut enc = Vec::with_capacity(norm_hist.write_bound());
        norm_hist.write(&mut enc).unwrap();
        let (dec_hist, rem) = NormHistogram::read(&enc).unwrap();
        assert!(rem.len() <= 1);
        assert_eq!(norm_hist, dec_hist);
    }

    #[test]
    fn uniform_dist_256() {
        // completely flat distribution
        for log2 in 8..=TABLE_LOG_MAX {
            let size = 1usize << log2;
            let mut data = Vec::with_capacity(size);
            for x in 0..=255u8 {
                for _ in 0..(1 << (log2 - 8)) {
                    data.push(x);
                }
            }
            let hist = Histogram::new(&data);
            for (j, &x) in hist.table().iter().enumerate() {
                assert_eq!(
                    x,
                    1 << (log2 - 8),
                    "symbol {} should be {} but is {}",
                    j,
                    1 << (log2 - 8),
                    x
                );
            }
            hist_verify(hist, log2);
        }
    }

    #[test]
    fn exp_dist() {
        for log2 in 8..=TABLE_LOG_MAX {
            let size = 1usize << log2;
            let mut remaining = size;
            let mut data = Vec::with_capacity(size);
            let mut sym = 0;
            loop {
                for _ in 0..(remaining >> 1) {
                    data.push(sym);
                }
                remaining -= remaining >> 1;
                sym += 1;
                if remaining == 1 {
                    data.push(sym);
                    break;
                }
            }
            let hist = Histogram::new(&data);
            for (j, &x) in hist.table().iter().enumerate() {
                use std::cmp::Ordering;
                let j = j as u32;
                let expected = match j.cmp(&log2) {
                    Ordering::Less => (1 << log2) >> (1 + j),
                    Ordering::Equal => 1,
                    Ordering::Greater => 0,
                };
                assert_eq!(
                    x, expected,
                    "symbol {} should be {} but is {}",
                    j, expected, x
                );
            }
            hist_verify(hist, log2);
        }
    }

    #[test]
    fn rand_dist_uniform() {
        let mut rng = rand::thread_rng();
        for log2 in 8..=TABLE_LOG_MAX {
            let size = 1usize << (log2 + 2);
            let mut data: Vec<u8> = vec![0u8; size];
            for _ in 0..8 {
                data.try_fill(&mut rng).unwrap();
                let hist = Histogram::new(&data);
                hist_verify(hist, log2);
            }
        }
    }
}
