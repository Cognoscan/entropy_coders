const BITS: usize = usize::BITS as usize;
const BYTES: usize = BITS / 8;
const HALF_BYTES: usize = BYTES / 2;
const HALF_BITS: usize = BITS / 2;

#[allow(dead_code)]
const MASK: [usize; 33] = [
    0,
    0x1,
    0x3,
    0x7,
    0xF,
    0x1F,
    0x3F,
    0x7F,
    0xFF,
    0x1FF,
    0x3FF,
    0x7FF,
    0xFFF,
    0x1FFF,
    0x3FFF,
    0x7FFF,
    0xFFFF,
    0x1_FFFF,
    0x3_FFFF,
    0x7_FFFF,
    0xF_FFFF,
    0x1F_FFFF,
    0x3F_FFFF,
    0x7F_FFFF,
    0xFF_FFFF,
    0x1FF_FFFF,
    0x3FF_FFFF,
    0x7FF_FFFF,
    0xFFF_FFFF,
    0x1FFF_FFFF,
    0x3FFF_FFFF,
    0x7FFF_FFFF,
    0xFFFF_FFFF,
];

#[inline]
fn find_mask(val: usize) -> usize {
    (1 << val) - 1
}

/// A tool for writing out a bit sequence.
#[derive(Debug)]
pub struct BitStackWriter<'a> {
    writer: &'a mut Vec<u8>,
    ptr: *mut u8,
    end_ptr: *const u8,
    storage: usize,
    bits: usize,
    initial_len: usize,
}

impl<'a> BitStackWriter<'a> {
    /// Create a new writer.
    pub fn new(writer: &'a mut Vec<u8>) -> Self {
        writer.reserve(2 * BYTES);
        let initial_len = writer.len();
        let spare = writer.spare_capacity_mut().as_mut_ptr_range();
        let ptr = spare.start as *mut u8;
        let end_ptr = unsafe {
            spare
                .end
                .offset(spare.end.align_offset(HALF_BYTES) as isize - (HALF_BYTES as isize))
        } as *const u8;
        Self {
            writer,
            ptr,
            end_ptr,
            storage: 0,
            bits: 0,
            initial_len,
        }
    }

    #[inline]
    pub fn flush(&mut self) {
        if self.ptr.align_offset(HALF_BYTES) != 0 {
            // We're not aligned yet, so now it's time to dump some bytes
            // We'll only write enough to bring us into alignment
            let to_write = (self.bits / 8) & (HALF_BYTES - 1);
            let bytes = self.storage.to_le_bytes();
            // SAFETY:
            // The pointer math is guaranteed to be safe because we first made
            // sure the Vec this points to had at least enough space for
            // aligning to a full word and then writing one more full word. The
            // writes are fine because we are looking at a non-null pointer, are
            // always aligned because these are 1-byte writes, and we're just
            // manipulating plain byte values.
            unsafe {
                self.ptr.write(bytes[0]);
                if HALF_BYTES > 2 {
                    self.ptr.add(1).write(bytes[1]);
                }
                if HALF_BYTES > 3 {
                    self.ptr.add(2).write(bytes[2]);
                }
                self.ptr = self.ptr.add(to_write);
                self.storage >>= to_write * 8;
                self.bits -= to_write * 8;
            }
            // If this didn't align us, return early.
            if self.ptr.align_offset(HALF_BYTES) != 0 {
                return;
            }
        }
        // SAFETY: A few separate unsafe things are happening here, because
        // we're directly manipulating raw pointers:
        // - We write to the Vec's pointer directly as an aligned write. This is
        //   OK because after every write, we make sure we haven't gone past the
        //   end yet, and the immediately preceeding code guarantees we'll be
        //   aligned once we reach this point.
        // - We increment the pointer by a halfword, which is always valid as
        //   we're just walking the pointer up towards the upper limit.
        // - If we need to reallocate, we need to recalcalate the pointers,
        //   which is just as safe as it was when we did it in the `new`
        //   function.
        unsafe {
            let inc = ((self.bits / HALF_BITS) != 0) as usize;
            self.raw_write();
            self.storage >>= inc * HALF_BITS;
            self.bits -= inc * HALF_BITS;
            self.ptr = self.ptr.add(inc * HALF_BYTES);
            // Handle a full buffer
            if self.ptr.offset_from(self.end_ptr) == 0 {
                self.writer
                    .set_len(self.ptr.offset_from(self.writer.as_ptr()) as usize);
                self.writer.reserve(self.writer.len() * 2);
                let spare = self.writer.spare_capacity_mut().as_mut_ptr_range();
                self.ptr = spare.start as *mut u8;
                self.end_ptr = spare
                    .end
                    .offset(spare.end.align_offset(HALF_BYTES) as isize - (HALF_BYTES as isize))
                    as *const u8;
            }
        }
    }

    /// Directly write a halfword to the internal Vec
    #[cfg(target_pointer_width = "64")]
    #[inline]
    unsafe fn raw_write(&mut self) {
        self.ptr
            .cast::<u32>()
            .write((self.storage.to_le() & 0xFFFF_FFFF) as u32);
    }

    /// Directly write a halfword to the internal Vec
    #[cfg(target_pointer_width = "32")]
    #[inline]
    unsafe fn raw_write(&mut self) {
        self.ptr.cast::<u16>().write((self.storage & 0xFFFF) as u16);
    }

    /// Write up to 16 bits directly to the internal buffer without flushing
    /// the internal buffer to the output stream. The bits to be written are
    /// masked before writing.
    ///
    /// # Safety
    /// This can cause data loss unless [`flush`] is run at at least every time
    /// up to 1/2 of the total internal buffer's size has been written - on
    /// 32-bit architectures, this means flushing every 16 potential bits, and
    /// on 64-bit architectures, this means flushing every 32 potential bits.
    ///
    #[inline]
    pub unsafe fn write_bits_raw_unmasked(&mut self, val: usize, bits: usize) {
        // Check for validity only in debug mode
        debug_assert!(
            bits <= 16,
            "Can only write up to 16 bits at a time, but tried to write {} bits",
            bits
        );
        let val = val & find_mask(bits);
        self.write_bits_raw(val, bits)
    }

    /// Write up to 16 bits directly to the internal buffer without flushing
    /// the internal buffer to the output stream. The bits to be written are the
    /// only ones allowed to be set to 1, all upper bits must be zero.
    ///
    /// # Safety
    /// This can cause data loss unless [`flush`] is run at at least every time
    /// up to 1/2 of the total internal buffer's size has been written - on
    /// 32-bit architectures, this means flushing every 16 potential bits, and
    /// on 64-bit architectures, this means flushing every 32 potential bits.
    ///
    #[inline]
    pub unsafe fn write_bits_raw(&mut self, val: usize, bits: usize) {
        // Check for validity only in debug mode
        debug_assert!(
            bits <= 16,
            "Can only write up to 16 bits at a time, but tried to write {} bits",
            bits
        );
        debug_assert!(
            (val & !((1 << bits) - 1)) == 0,
            "Unused bits in `val` are nonzero. Val = 0x{:x}, bits = {}",
            val,
            bits
        );
        // OR in the bits and update the state
        self.storage |= val << self.bits;
        self.bits += bits;
    }

    /// Write up to 16 bits into the output stream. Assumes that `bits` is 16 or
    /// less, and that the unused bits in `val` are 0.
    #[inline]
    pub fn write_bits(&mut self, val: usize, bits: usize) {
        unsafe {
            self.write_bits_raw(val, bits);
        }
        self.flush();
    }

    /// Write up to 16 bits into the output stream. Assumes that `bits` is 16 or
    /// less.
    #[inline]
    pub fn write_bits_unmasked(&mut self, val: usize, bits: usize) {
        unsafe { self.write_bits_raw_unmasked(val, bits) }
        self.flush();
    }

    /// Finish writing to the stack and return the number of bits written.
    pub fn finish(mut self) -> usize {
        // Double-flush to push out all the remaining bits. We pretend we have a
        // completely full buffer to make sure we get everything out.
        let actual_bits = self.bits;
        self.bits = BITS;
        self.flush();
        self.flush();
        // Correct the vec to have the right length
        let total_size = unsafe { self.ptr.offset_from(self.writer.as_ptr()) as usize };
        let total_bits = total_size * 8 + actual_bits - BITS;
        let new_len = (total_bits + 7) / 8;
        unsafe {
            self.writer.set_len(new_len);
        }
        // Return the total bytes written to the output
        total_bits - (self.initial_len * 8)
    }
}

/// A tool for reading bits off a stack.
#[derive(Clone, Debug)]
pub struct BitStackReader<'a> {
    reader: &'a [u8],
    ptr: *const u8,
    bits: usize,
    buffer: usize,
    finished: bool,
}

impl<'a> BitStackReader<'a> {
    /// Create a new bit reader. This assumes there is a final marker bit set to
    /// 1 to indicate the end of the bit stack, and fails to initialize if that
    /// marker bit is missing.
    pub fn new(reader: &'a [u8]) -> Option<Self> {
        if reader.is_empty() {
            return None;
        }
        // Load our initial pointer. If we can align it now, we will do so.
        //
        // SAFETY: We just verified the reader isn't empty, so the end pointer
        // can be rewound at least one byte. When we try to align the pointer,
        // we also make sure to check that doing so won't put us past the start
        // of the allocation.
        let mut ptr = unsafe {
            let ptr = reader.as_ptr_range().end.offset(-1);
            // Try to align the buffer
            let align = ptr.align_offset(HALF_BYTES);
            // We can't even accidentally go past the start of the pointer, so check before we do
            if ptr.offset_from(reader.as_ptr()) > (HALF_BYTES - align) as isize {
                ptr.offset(align as isize - (HALF_BYTES as isize))
            } else {
                reader.as_ptr()
            }
        };

        // Manually read out the first few bytes
        let to_read = reader.len() - (unsafe { ptr.offset_from(reader.as_ptr()) } as usize);
        let mut bytes = [0; BYTES];
        for (i, b) in bytes.iter_mut().take(to_read).enumerate() {
            *b = unsafe { ptr.add(i).read() };
        }
        let buffer: usize = usize::from_le_bytes(bytes);
        let bits = to_read * 8;
        let finished = ptr == reader.as_ptr();
        // Try to move the pointer down
        unsafe {
            if ptr.offset_from(reader.as_ptr()) >= (HALF_BYTES as isize) {
                ptr = ptr.offset(-(HALF_BYTES as isize));
            } else {
                ptr = reader.as_ptr();
            }
        }
        let mut this = Self {
            reader,
            ptr,
            bits,
            buffer,
            finished,
        };

        // Do a standard reload in order to fully populate the stream
        this.reload();

        // With our bits read, it's time to figure out how many bits are
        // actually present, by reading the marker bit at the end of the stream.
        // If the slice had completely empty bytes at the end, we should fail
        // and return None, as that means a framing error has occurred.
        if this.buffer == 0 {
            return None;
        }
        let highbit = this.buffer.ilog2() as usize;
        if (this.bits - highbit) > 8 {
            return None;
        }

        this.bits = highbit;
        this.reload();
        Some(this)
    }

    /// Replenish the internal buffer. Ensures that the internal buffer has at
    /// least usize::BITS/2 bits left inside it, provided there were more bits
    /// to retrieve.
    pub fn reload(&mut self) {
        // If we're finished, there's nothing to do.
        if self.finished {
            return;
        }

        // Final readout - we're at the base now
        if self.ptr == self.reader.as_ptr() {
            // This reads however many bytes are left between the base pointer
            // and the next higher aligned value we would've previously read
            // from.
            let to_read = HALF_BYTES - ((self.ptr as usize) & (HALF_BYTES - 1));
            // If we're actually reading bits this time around, then we're going
            // to be finished.
            self.finished = self.bits <= HALF_BITS;
            // This mask ensures we actually *don't* have any effect if we're
            // over half full (and thus may not have space to take on these
            // extra bits)
            let mask = self.finished as usize * usize::MAX;
            let to_read = to_read & mask;

            // Read the bits
            let mut read_bytes = [0; HALF_BYTES];
            for (i, b) in read_bytes.iter_mut().take(to_read).enumerate() {
                *b = unsafe { self.ptr.add(i).read() };
            }
            #[cfg(target_pointer_width = "32")]
            let read = u16::from_le_bytes(read_bytes);
            #[cfg(target_pointer_width = "64")]
            let read = u32::from_le_bytes(read_bytes);
            let read_bits = 8 * to_read;

            // Load the bits we read up into the the buffer. This time we don't
            // need to mask the read part because it'll be 0 if we had `to_read`
            // set to zero.
            self.buffer = (self.buffer << read_bits) | (read as usize);
            self.bits += read_bits;

            return;
        }

        let will_read = (self.bits <= HALF_BITS) as usize;
        let read_bits = will_read * HALF_BITS;
        let read_mask = will_read * usize::MAX;

        // Read in the bits, every time
        // SAFETY: If we made it this far, then we're aligned (courtesy of the
        // code in `new`), and we have at least HALF_BYTES available, as that's
        // how many we would have subtracted our pointer by.
        #[cfg(target_pointer_width = "32")]
        let read = unsafe { self.ptr.cast::<u16>().read() };
        #[cfg(target_pointer_width = "64")]
        let read = unsafe { self.ptr.cast::<u32>().read() };

        // Load the bits we read up into the buffer.
        self.buffer = (self.buffer << read_bits) | ((read as usize) & read_mask);
        self.bits += read_bits;

        // Update the pointer, clamping to the base of the slice
        self.ptr = if unsafe { self.ptr.offset_from(self.reader.as_ptr()) } >= (HALF_BYTES as isize)
        {
            unsafe { self.ptr.offset(-((will_read * HALF_BYTES) as isize)) }
        } else {
            self.reader.as_ptr()
        };
    }

    /// Peek at the next N bits in the buffer, up to 16. Fails if there aren't
    /// enough bits left in the buffer.
    pub fn peek(&self, bits: usize) -> Option<usize> {
        debug_assert!(bits <= 16, "Can't read more than 16 bits at a time");
        if bits > self.bits {
            return None;
        }
        Some((self.buffer >> (self.bits - bits)) & find_mask(bits))
    }

    /// Read bits out without reloading the buffer.
    ///
    /// # Safety
    /// Only use this when you know the buffer should still have enough bits
    /// left for your read (unless the buffer holds all remaining bits in the
    /// stream).
    ///
    pub unsafe fn read_no_reload(&mut self, bits: usize) -> Option<usize> {
        let read = self.peek(bits)?;
        self.bits -= bits;
        Some(read)
    }

    /// Advance the buffer manually, without reloading or checking for bit validity.
    ///
    /// # Safety
    /// `bits` must be less than the number of remaining bits in the buffer, and
    /// subsequent reads shouldn't deplete the remaining bits in the buffer.
    pub unsafe fn advance_no_reload(&mut self, bits: usize) {
        debug_assert!(bits <= self.bits);
        self.bits -= bits;
    }

    /// Read the next N bits in the buffer, up to 16. Fails if there aren't
    /// enough bits left in the buffer.
    pub fn read(&mut self, bits: usize) -> Option<usize> {
        let read = unsafe { self.read_no_reload(bits)? };
        self.reload();
        Some(read)
    }

    /// Get how many bits are left in the internal buffer
    pub fn available(&self) -> usize {
        self.bits
    }

    /// Finish reading. Returns whether or not the slice was completely read out
    /// or not.
    pub fn finish(self) -> bool {
        self.finished && self.bits == 0
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
        let out = (word >> bit_offset) & ((1 << bits) - 1);
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

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    fn encode(test_vec: &[(usize, usize)], mark: bool) -> (Vec<u8>, usize) {
        // Encode stage
        let mut encoded = Vec::new();
        let mut enc = BitStackWriter::new(&mut encoded);
        let mut total_bits = 0;
        for (val, bits) in test_vec {
            total_bits += bits;
            enc.write_bits(*val, *bits);
        }
        let written_bits = if mark {
            enc.write_bits(1, 1);
            enc.finish() - 1
        } else {
            enc.finish()
        };
        assert_eq!(
            total_bits, written_bits,
            "Writer didn't actually write as many bits as we told it to"
        );
        let total_bytes = (total_bits + mark as usize + 7) / 8;
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

    fn decode_stack(encoded: &[u8], test_vec: &[(usize, usize)]) {
        // Decode as a stack
        let mut dec = BitStackReader::new(encoded).unwrap();
        println!("dec.buffer = 0x{:x}", dec.buffer);
        println!("vec has {} pairs", test_vec.len());
        for (i, (val, bits)) in test_vec.iter().enumerate().rev() {
            let read_val = dec.read(*bits).unwrap_or_else(|| {
                println!("dec.buffer = 0x{:x}", dec.buffer);
                panic!(
                    "Bitstack: Should have been able to read bits, failed on bit {} from start",
                    i
                )
            });
            assert_eq!(
                read_val, *val,
                "Bitstack: Expected to get 0x{}, got 0x{}",
                val, read_val
            );
        }
        let bits_in_buf = dec.available();
        assert!(dec.finish(), "Decoder wasn't finished");
        assert!(
            bits_in_buf == 0,
            "Decoder still had bits left inside the internal buffer"
        );
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
        let (encoded, bits_left, offset) = dec.finish();
        assert!(encoded.len() <= 1);
        assert!(bits_left == 0);
        assert!(offset <= 8);
    }

    #[test]
    fn stack_tests() {
        let mut test_vec = Vec::new();
        for i in 0..(BITS * 5) {
            test_vec.push((i & 0x1, 1));
            let (encoded, _) = encode(&test_vec, true);
            decode_stack(&encoded, &test_vec);
        }

        // Repeat a few more times with random bits
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Standard;
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..(BITS * 5) {
                let v: bool = rng.sample(dist);
                test_vec.push((v as usize, 1));
                let (encoded, _) = encode(&test_vec, true);
                decode_stack(&encoded, &test_vec);
            }
        }

        // Repeat again, this time with random numbers of bits
        let dist_val = rand::distributions::Standard;
        let dist_bits = rand::distributions::Uniform::new_inclusive(1, 16);
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..100 {
                let bits: usize = rng.sample(dist_bits);
                let val: usize = rng.sample(dist_val);
                let val = val & ((1 << bits) - 1);
                test_vec.push((val, bits));
                let (encoded, _) = encode(&test_vec, true);
                decode_stack(&encoded, &test_vec);
            }
        }
    }

    #[test]
    fn stream_tests() {
        let mut test_vec = Vec::new();
        for i in 0..(BITS * 2) {
            test_vec.push((i & 0x1, 1));
            let (encoded, total_bits) = encode(&test_vec, false);
            decode_stream(&encoded, total_bits, &test_vec);
        }

        // Repeat a few more times with random bits
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Standard;
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..(BITS * 5) {
                let v: bool = rng.sample(dist);
                test_vec.push((v as usize, 1));
                let (encoded, total_bits) = encode(&test_vec, false);
                decode_stream(&encoded, total_bits, &test_vec);
            }
        }

        // Repeat again, this time with random numbers of bits
        let dist_val = rand::distributions::Standard;
        let dist_bits = rand::distributions::Uniform::new_inclusive(1, 16);
        for _ in 0..10 {
            test_vec.clear();
            for _ in 0..100 {
                let bits: usize = rng.sample(dist_bits);
                let val: usize = rng.sample(dist_val);
                let val = val & ((1 << bits) - 1);
                test_vec.push((val, bits));
                let (encoded, total_bits) = encode(&test_vec, false);
                decode_stream(&encoded, total_bits, &test_vec);
            }
        }
    }
}
