
pub struct Histogram {
    table: [u32; 256],
    size: usize,
}

impl Histogram {

    /// Create a new histogram
    pub fn new(data: &[u8]) -> Self {
        let mut table = [0u32; 256];
        for i in data {
            // SAFETY: i is a u8 and thus 0-255, and our table is 256 long. It
            // will *never* be out of range.
            unsafe {
                *(table.get_unchecked_mut(*i as usize)) += 1;
            }
        }
        Self {
            table,
            size: data.len()
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

    /// Get the number of symbols that were in the original data
    pub fn size(&self) -> usize {
        self.size
    }
}