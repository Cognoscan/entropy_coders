use std::io::Write;

const BITS: usize = usize::BITS as usize;
const BYTES: usize = BITS / 8;

/// A tool for writing out a bit sequence.
///
/// This is meant for use with `Vec<u8>` and `&mut [u8]`. Using this on an
/// unbuffered I/O resource is a recipe for a bad time.
#[derive(Debug)]
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

    // Flush `to_write` bits of data to the writer. Safe only if `to_write` is
    // less than or equal to usize::BITS.
    fn flush(&mut self, to_write: usize) -> std::io::Result<()> {
        debug_assert!(to_write <= BITS);
        // Flush existing bits. Most of the time, we're not actually flushing
        // bits at all, but this should produce less branch mis-predicts than if
        // we wrapped this in an if statement.
        let to_write_bytes = (to_write + 7) / 8;
        let bytes = self.storage.to_le_bytes();
        let bytes_write: &[u8] = unsafe { bytes.get_unchecked(0..to_write_bytes) };
        let written = self.writer.write(bytes_write)?;
        if written != to_write_bytes {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        self.bits -= to_write;
        self.storage >>= to_write;
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

        // We write either the half the word, or 0. If we mask out the portion
        // of the bit offset that indexes within a halfword, we'll just get a
        // number of bits to write that's either a halfword's worth, or 0.
        let to_write = self.bits & !(HALF_BITS - 1);
        self.flush(to_write)?;

        // OR in the bits and update the state
        self.storage |= val << self.bits;
        self.bits += bits;
        self.written += bits;
        Ok(())
    }

    /// Finish writing to the stack and return the number of bits written.
    pub fn finish(mut self) -> std::io::Result<usize> {
        self.flush(self.bits)?;
        Ok(self.written)
    }
}

/// A tool for reading bits off a stack.
#[derive(Clone, Debug)]
pub struct BitStackReader<'a> {
    reader: &'a [u8],
    available: usize,
    last0: usize,
    last1: usize,
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

        let mut word_offset = ((total_bits-1) / BITS) * BYTES;
        let mut last0 = 0;
        for i in (0..BYTES).rev() {
            if let Some(val) = reader.get(word_offset + i) {
                last0 = (last0 << 8) | (*val as usize);
            }
        }

        if word_offset + (BYTES/2) > reader.len() {
            word_offset = word_offset.saturating_sub(BYTES/2);
        }
        else {
            word_offset += BYTES/2;
        }
        let mut last1 = 0;
        for i in (0..BYTES).rev() {
            if let Some(val) = reader.get(word_offset + i) {
                last1 = (last1 << 8) | (*val as usize);
            }
        }
        Self {
            reader,
            available: total_bits,
            last0,
            last1
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
            if (idx / (BYTES/2)) & 1 == 1 {
                self.last1
            }
            else {
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

        let mut word_offset = ((total_bits-1) / BITS) * BYTES;
        let mut last0 = 0;
        for i in (0..BYTES).rev() {
            if let Some(val) = reader.get(word_offset + i) {
                last0 = (last0 << 8) | (*val as usize);
            }
        }

        if word_offset + (BYTES/2) > reader.len() {
            word_offset = word_offset.saturating_sub(BYTES/2);
        }
        else {
            word_offset += BYTES/2;
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
            last1
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
            if (idx / (BYTES/2)) & 1 == 1 {
                self.last1
            }
            else {
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

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    fn encode(test_vec: &[(usize, usize)]) -> (Vec<u8>, usize) {
        // Encode stage
        let mut encoded = Vec::new();
        let mut enc = BitStackWriter::new(&mut encoded);
        let mut total_bits = 0;
        for (val, bits) in test_vec {
            total_bits += bits;
            enc.write_bits(*val, *bits)
                .expect("Should always be able to write to a Vec");
        }
        let written_bits = enc
            .finish()
            .expect("Should always be able to write to a Vec");
        assert_eq!(
            total_bits, written_bits,
            "Writer didn't actually write as many bits as we told it to"
        );
        let total_bytes = (total_bits + 7) / 8;
        assert_eq!(
            encoded.len(),
            total_bytes,
            "Number of bytes in vec ({}) isn't as expected ({})",
            encoded.len(),
            total_bytes
        );

        if encoded.len() <= 64 {
            println!("test_vec: {:?}", test_vec);
            println!("encoded (hex): {:x?}", encoded);
        }
        (encoded, total_bits)
    }

    fn decode_stack(encoded: &[u8], total_bits: usize, test_vec: &[(usize, usize)]) {
        // Decode as a stack
        let mut dec = BitStackReader::new(encoded, total_bits);
        for (val, bits) in test_vec.iter().rev() {
            let read_val = dec
                .read(*bits)
                .expect("Bitstack: Should have been able to read bits");
            assert_eq!(
                read_val, *val,
                "Bitstack: Expected to get 0x{}, got 0x{}",
                val, read_val
            );
        }
    }

    fn decode_stream(encoded: &[u8], total_bits: usize, test_vec: &[(usize, usize)]) {
        // Decode as a stream
        let mut dec = BitStreamReader::new(encoded, total_bits);
        for (val, bits) in test_vec {
            let read_val = dec
                .read(*bits)
                .expect("Bitstream: Should have been able to read bits");
            assert_eq!(
                read_val, *val,
                "Bitstream: Expected to get 0x{}, got 0x{}",
                val, read_val
            );
        }
    }

    #[test]
    fn stack_tests() {
        let mut test_vec = Vec::new();
        for i in 0..(BITS * 5) {
            test_vec.push((i & 0x1, 1));
            let (encoded, total_bits) = encode(&test_vec);
            decode_stack(&encoded, total_bits, &test_vec);
        }

        // Repeat a few more times with random bits
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Standard;
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..(BITS*5) {
                let v: bool = rng.sample(dist);
                test_vec.push((v as usize, 1));
                let (encoded, total_bits) = encode(&test_vec);
                decode_stack(&encoded, total_bits, &test_vec);
            }
        }

        // Repeat again, this time with random numbers of bits
        let dist_val = rand::distributions::Standard;
        let dist_bits = rand::distributions::Uniform::new_inclusive(1,16);
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..100 {
                let bits: usize = rng.sample(dist_bits);
                let val: usize = rng.sample(dist_val);
                let val = val & ((1<<bits)-1);
                test_vec.push((val, bits));
                let (encoded, total_bits) = encode(&test_vec);
                decode_stack(&encoded, total_bits, &test_vec);
            }
        }
    }

    #[test]
    fn stream_tests() {
        let mut test_vec = Vec::new();
        for i in 0..(BITS * 2) {
            test_vec.push((i & 0x1, 1));
            let (encoded, total_bits) = encode(&test_vec);
            decode_stream(&encoded, total_bits, &test_vec);
        }

        // Repeat a few more times with random bits
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Standard;
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..(BITS*5) {
                let v: bool = rng.sample(dist);
                test_vec.push((v as usize, 1));
                let (encoded, total_bits) = encode(&test_vec);
                decode_stream(&encoded, total_bits, &test_vec);
            }
        }

        // Repeat again, this time with random numbers of bits
        let dist_val = rand::distributions::Standard;
        let dist_bits = rand::distributions::Uniform::new_inclusive(1,16);
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..100 {
                let bits: usize = rng.sample(dist_bits);
                let val: usize = rng.sample(dist_val);
                let val = val & ((1<<bits)-1);
                test_vec.push((val, bits));
                let (encoded, total_bits) = encode(&test_vec);
                decode_stream(&encoded, total_bits, &test_vec);
            }
        }
    }
}
