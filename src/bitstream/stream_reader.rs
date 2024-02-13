use super::*;

/// A tool for reading bits off a stream
#[derive(Clone, Debug)]
pub struct BitStreamReader<'a> {
    reader: &'a [u8],
    total_bits: usize,
    bits_read: usize,
    last0: usize,
    last1: usize,
}

impl<'a> BitStreamReader<'a> {
    /// Create a new bit reader. The slice provided should be exactly the
    /// required number of bytes to contain the total number of bits stated.
    pub fn new(reader: &'a [u8], total_bits: usize) -> Self {
        assert!(!reader.is_empty(), "No bytes provided to read from");
        assert!(
            ((total_bits + 7) / 8) == reader.len(),
            "Total number of bytes should be exactly enough to contain the total number of bits"
        );

        let mut word_offset = ((total_bits - 1) / BITS) * BYTES;
        let mut last0 = 0;
        for i in (0..BYTES).rev() {
            if let Some(val) = reader.get(word_offset + i) {
                last0 = (last0 << 8) | (*val as usize);
            }
        }

        if word_offset + (BYTES / 2) > reader.len() {
            word_offset = word_offset.saturating_sub(BYTES / 2);
        } else {
            word_offset += BYTES / 2;
        }
        let mut last1 = 0;
        for i in (0..BYTES).rev() {
            if let Some(val) = reader.get(word_offset + i) {
                last1 = (last1 << 8) | (*val as usize);
            }
        }
        Self {
            reader,
            total_bits,
            bits_read: 0,
            last0,
            last1,
        }
    }

    /// Pull some bits off the buffer.
    ///
    /// `bits` must be greater than 0 and less than or equal to usize::BITS/2.
    /// This is only checked in debug mode, as this is expected to be part of a
    /// tight loop in regular code.
    pub fn read(&mut self, bits: usize) -> std::io::Result<usize> {
        let ret = self.peek(bits)?;
        self.advance_by(bits)?;
        Ok(ret)
    }

    /// Advance the buffer pointer by N bits.
    ///
    /// `bits` must be greater than 0 and less than or equal to usize::BITS/2.
    /// This is only checked in debug mode, as this is expected to be part of a
    /// tight loop in regular code.
    pub fn advance_by(&mut self, bits: usize) -> std::io::Result<()> {
        debug_assert!(bits <= BITS / 2);
        debug_assert!(bits != 0);
        if self.bits_read + bits > self.total_bits {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        self.bits_read += bits;
        Ok(())
    }

    /// Peek at bits in the buffer.
    ///
    /// `bits` must be greater than 0 and less than or equal to usize::BITS/2.
    /// This is only checked in debug mode, as this is expected to be part of a
    /// tight loop in regular code.
    pub fn peek(&self, bits: usize) -> std::io::Result<usize> {
        debug_assert!(bits <= BITS / 2);
        debug_assert!(bits != 0);
        if self.bits_read + bits > self.total_bits {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }

        // Set up the halfword/bit indices.
        let idx = (self.bits_read / (BITS / 2)) * (BYTES / 2);
        let bit_offset = self.bits_read & (BITS / 2 - 1);

        let word = if idx + BYTES > self.reader.len() {
            if (idx / (BYTES / 2)) & 1 == 1 {
                self.last1
            } else {
                self.last0
            }
        } else {
            // SAFETY: the earlier check should ensure that this code doesn't
            // run unless we have at least BYTES available in the slice starting
            // at `idx`. The cast should be safe because we literally just
            // grabbed a slice of exactly size BYTES.
            let bytes: &[u8; BYTES] = unsafe {
                let bytes = self.reader.get_unchecked(idx..(idx + BYTES));
                &*(bytes.as_ptr().cast::<[u8; BYTES]>())
            };
            usize::from_le_bytes(*bytes)
        };

        // Apply the bit offset and mask just the desired bits.
        let out = (word >> bit_offset) & find_mask(bits);
        Ok(out)
    }

    /// Get how many bits are left in the buffer.
    pub fn available(&self) -> usize {
        self.total_bits - self.bits_read
    }

    /// Finish reading. Returns the remaining bits as a slice, the number of
    /// bits left in the slice, and the bit offset into the first byte.
    pub fn finish(self) -> (&'a [u8], usize, usize) {
        let remaining = self.total_bits - self.bits_read;
        let byte = self.bits_read / 8;
        let offset = self.bits_read % 8;
        (&self.reader[byte..], remaining, offset)
    }

    /// Finish reading, completing the current byte and returning whatever bytes
    /// are left.
    pub fn finish_byte(self) -> &'a [u8] {
        let byte = (self.bits_read + 7) / 8;
        &self.reader[byte..]
    }
}
