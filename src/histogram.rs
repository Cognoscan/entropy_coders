/// A histogram of all the 8-bit symbols seen in a data sequence.
#[derive(Clone, Debug)]
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

    /// Normalize the histogram. Panics if log2 is not between 9 and 16.
    pub fn normalize(self, log2: u32) -> NormHistogram {
        assert!(
            (9..=16).contains(&log2),
            "Histogram normalization: log2 must be between 9 & 16, but is {}",
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

        for (i, (&t, t_norm)) in self
            .table
            .iter()
            .zip(table.iter_mut())
            .take(self.max_symbol as usize)
            .enumerate()
        {
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
                prob += (t as u64 * step) - (((prob << scale) > rest_to_beat) as u64);
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
        if -to_distribute >= (largest_prob >> 1) {
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
        const UNASSIGNED: i32 = -2;
        let low_threshold = self.size >> log2;
        let low_one = (self.size * 3) >> (log2 + 1);
        let mut table = [0i32; 256];
        let mut to_distribute = 1 << (log2 as i32);
        let mut total = self.size as u32;

        // Begin by distributing the low-probability symbols
        for (&t, t_norm) in self
            .table
            .iter()
            .zip(table.iter_mut())
            .take(self.max_symbol as usize)
        {
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
                .take(self.max_symbol as usize)
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
        }
        else if total == 0 {
            // If all values are pretty poor, just evenly spread the values across
            // the remainder
            while to_distribute != 0 {
                // Repeat until we're out of numbers to distribute.
                for t in table.iter_mut().take(self.max_symbol as usize) {
                    if *t > 0 {
                        *t += 1;
                        to_distribute -= 1;
                        if to_distribute == 0 {
                            break;
                        }
                    }
                }
            }
        }
        else {
            let v_step_log = 62 - log2;
            let mid = (1 << (v_step_log-1)) - 1;
            let r_step = (((1 << v_step_log) * to_distribute) + mid) / total;
            let mut tmp_total = mid;
            for (&t, t_norm) in self
                .table
                .iter()
                .zip(table.iter_mut())
                .take(self.max_symbol as usize)
            {
                if *t_norm == UNASSIGNED {
                    let end = tmp_total + (t * r_step);
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
}
