use super::*;

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
        // Find the last aligned pointer that's still inside the Vec (or 1 past the end).
        // SAFETY: We should be able to do this because we had previously
        // reserved at least 2*BYTES bytes in the vec.
        let end_ptr = unsafe {
            let align = spare.end.align_offset(HALF_BYTES);
            if align != 0 {
                spare.end.offset(align as isize - (HALF_BYTES as isize))
            } else {
                spare.end
            }
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
        let align = self.ptr.align_offset(HALF_BYTES);
        if align != 0 {
            // We're not aligned yet, so now it's time to dump some bytes
            // We'll only write enough to bring us into alignment
            let to_write = (self.bits / 8).min(align);
            debug_assert!(to_write < HALF_BYTES);
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
            }
            self.storage >>= to_write * 8;
            self.bits -= to_write * 8;
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
            if (self.ptr as *const u8) == self.end_ptr {
                self.writer
                    .set_len(self.ptr.offset_from(self.writer.as_ptr()) as usize);
                self.writer.reserve(self.writer.len() * 2 + HALF_BYTES);
                let spare = self.writer.spare_capacity_mut().as_mut_ptr_range();
                self.ptr = spare.start as *mut u8;

                self.end_ptr = {
                    let align = spare.end.align_offset(HALF_BYTES);
                    if align != 0 {
                        spare.end.offset(align as isize - (HALF_BYTES as isize))
                    } else {
                        spare.end
                    }
                } as *const u8;
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
    /// This can cause data loss unless [`flush`][BitStackWriter::flush] is run
    /// at least every time up to 1/2 of the total internal buffer's size has
    /// been written - on 32-bit architectures, this means flushing every 16
    /// potential bits, and on 64-bit architectures, this means flushing every
    /// 32 potential bits.
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
    /// This can cause data loss unless [`flush`][BitStackWriter::flush] is run
    /// at least every time up to 1/2 of the total internal buffer's size has
    /// been written - on 32-bit architectures, this means flushing every 16
    /// potential bits, and on 64-bit architectures, this means flushing every
    /// 32 potential bits.
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
        //println!("write: {:04x}, {} bits", val, bits);
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
        // Figure out how many bits we have in the buffer at the end
        let total_size = unsafe { self.ptr.offset_from(self.writer.as_ptr()) as usize };
        let actual_bits = self.bits;
        let total_bits = total_size * 8 + actual_bits;

        // Double-flush to push out all the remaining bits. We pretend we have a
        // completely full buffer to make sure we get everything out.
        self.bits = BITS;
        self.flush();
        self.bits = BITS;
        self.flush();

        // Correct the vec to have the right length
        //println!("total_size = {}, actual_bits = {}, total_bits = {}", total_size, actual_bits, total_bits);
        let new_len = (total_bits + 7) / 8;
        unsafe {
            self.writer.set_len(new_len);
        }
        // Return the total bytes written to the output
        total_bits - (self.initial_len * 8)
    }
}
