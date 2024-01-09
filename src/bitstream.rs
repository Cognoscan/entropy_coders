use std::io::Write;

const BITS: usize = usize::BITS as usize;
const BYTES: usize = BITS / 8;

/// A tool for writing out a bit sequence.
///
/// This is meant for use with `Vec<u8>` and `&mut [u8]`. Using this on an
/// unbuffered I/O resource is a recipe for a bad time.
pub struct BitStackWriter<'a, T: Write> {
    writer: &'a mut T,
    storage: usize,
    bits: usize,
    written: usize,
}

impl<'a, T: Write> BitStackWriter<'a, T> {
    /// Create a new writer.
    pub fn new(writer: &'a mut T) -> Self {
        Self {
            writer,
            storage: 0,
            bits: 0,
            written: 0,
        }
    }

    // Flush `to_write` bytes of data to the writer. Safe only if `to_write` is
    // less than or equal to usize::BITS/8.
    fn flush(&mut self, to_write: usize) -> std::io::Result<()> {
        debug_assert!(to_write <= BYTES);
        // Flush existing bits. Most of the time, we're not actually flushing
        // bits at all, but this should produce less branch mis-predicts than if
        // we wrapped this in an if statement.
        let bytes = self.storage.to_le_bytes();
        let bytes_write: &[u8] = unsafe { bytes.get_unchecked(0..to_write) };
        let written = self.writer.write(bytes_write)?;
        if written != to_write {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        let write_bits = to_write * 8;
        self.bits -= write_bits;
        self.storage >>= write_bits;
        Ok(())
    }

    /// Write up to 16 bits into the output stream. Assumes that `bits` is 16 or
    /// less, and that the unused bits in `val` are 0.
    pub fn write_bits(&mut self, val: usize, bits: usize) -> std::io::Result<()> {
        const HALF_BITS: usize = BITS / 2;
        // Check for validity only in debug mode
        debug_assert!(
            (val & !((1 << bits) - 1)) == 0,
            "Unused bits in `val` are nonzero. Val = 0x{:x}, bits = {}",
            val,
            bits
        );
        debug_assert!(
            bits <= 16,
            "Can only write up to 16 bits at a time, but tried to write {} bits",
            bits
        );

        // We write either the half the word, or 0. If we divide by HALF_BITS,
        // we'll get the number of half-words. Multiply that up by the number of
        // bytes a half-word and that's what we'll write.
        // Note: these divides/multiplies should become bitshifts.
        let to_write = (self.bits / HALF_BITS) * (HALF_BITS / 8);
        self.flush(to_write)?;

        // OR in the bits and update the state
        self.storage |= val << self.bits;
        self.bits += bits;
        self.written += bits;
        Ok(())
    }

    /// Finish writing to the stack and return the number of bits written.
    pub fn finish(mut self) -> std::io::Result<usize> {
        self.flush((self.bits + 7) / 8)?;
        Ok(self.written)
    }
}

/// A tool for reading bits off a stack.
pub struct BitStackReader<'a> {
    reader: &'a [u8],
    available: usize,
    last_word: usize,
}

impl<'a> BitStackReader<'a> {
    /// Create a new bit reader. The slice provided should be exactly the
    /// required number of bytes to contain the total number of bits stated.
    pub fn new(reader: &'a [u8], total_bits: usize) -> Self {
        assert!(!reader.is_empty(), "No bytes provided to read from");
        assert!(
            ((total_bits + 7) / 8) == reader.len(),
            "Total number of bytes should be exactly enough to contain the total number of bits"
        );

        let halfword_offset = (total_bits / (BITS / 2)) * (BYTES / 2);
        let word_offset = halfword_offset.saturating_sub(BYTES / 2);
        let mut last_word = 0;
        for i in (0..BYTES).rev() {
            if let Some(val) = reader.get(word_offset + i) {
                last_word = (last_word << 8) | (*val as usize);
            }
        }
        Self {
            reader,
            available: total_bits,
            last_word,
        }
    }

    /// Pop some bits off the buffer. `bits` must be greater than 0 and less
    /// than or equal to usize::BITS/2. This is only checked in debug mode, as
    /// this is expected to be part of a tight loop in regular code.
    pub fn read(&mut self, bits: usize) -> std::io::Result<usize> {
        debug_assert!(bits <= BITS / 2);
        debug_assert!(bits != 0);
        if self.available < bits {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        self.available -= bits;

        // Set up the halfword/bit indices.
        let idx = (self.available / (BITS / 2)) * (BYTES / 2);
        let bit_offset = self.available & (BITS / 2 - 1);

        let word = if idx + BYTES > self.reader.len() {
            if idx + (BYTES / 2) > self.reader.len() {
                self.last_word >> (BITS / 2)
            } else {
                self.last_word
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
        let out = (word >> bit_offset) & ((1 << bits) - 1);
        Ok(out)
    }

    /// Get how many bits are left in the buffer.
    pub fn available(&self) -> usize {
        self.available
    }

    /// Finish reading. Returns a failure if there are still bits left to read.
    pub fn finish(self) -> std::io::Result<()> {
        if self.available > 0 {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        Ok(())
    }
}

/// A tool for reading bits off a stream
pub struct BitStreamReader<'a> {
    reader: &'a [u8],
    total_bits: usize,
    last_word: usize,
    bits_read: usize,
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

        let halfword_offset = (total_bits / (BITS / 2)) * (BYTES / 2);
        let word_offset = halfword_offset.saturating_sub(BYTES / 2);
        let mut last_word = 0;
        for i in (0..BYTES).rev() {
            if let Some(val) = reader.get(word_offset + i) {
                last_word = (last_word << 8) | (*val as usize);
            }
        }
        Self {
            reader,
            total_bits,
            bits_read: 0,
            last_word,
        }
    }

    /// Pull some bits off the buffer. `bits` must be greater than 0 and less
    /// than or equal to usize::BITS/2. This is only checked in debug mode, as
    /// this is expected to be part of a tight loop in regular code.
    pub fn read(&mut self, bits: usize) -> std::io::Result<usize> {
        debug_assert!(bits <= BITS / 2);
        debug_assert!(bits != 0);
        if self.bits_read + bits > self.total_bits {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }

        // Set up the halfword/bit indices.
        let idx = (self.bits_read / (BITS / 2)) * (BYTES / 2);
        let bit_offset = self.bits_read & (BITS / 2 - 1);

        let word = if idx + BYTES > self.reader.len() {
            if idx + (BYTES / 2) > self.reader.len() {
                self.last_word >> (BITS / 2)
            } else {
                self.last_word
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

        // Update our offset
        self.bits_read += bits;

        // Apply the bit offset and mask just the desired bits.
        let out = (word >> bit_offset) & ((1 << bits) - 1);
        Ok(out)
    }

    /// Get how many bits are left in the buffer.
    pub fn available(&self) -> usize {
        self.total_bits - self.bits_read
    }

    /// Finish reading. Returns a failure if there are still bits left to read.
    pub fn finish(self) -> std::io::Result<()> {
        if self.total_bits != self.bits_read {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        Ok(())
    }
}
