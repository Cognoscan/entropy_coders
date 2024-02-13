use super::*;

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