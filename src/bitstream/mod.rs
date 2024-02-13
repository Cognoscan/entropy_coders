const BITS: usize = usize::BITS as usize;
const BYTES: usize = BITS / 8;
const HALF_BYTES: usize = BYTES / 2;
const HALF_BITS: usize = BITS / 2;

use crate::find_mask;

mod writer;
pub use writer::BitStackWriter;

mod stack_reader;
pub use stack_reader::BitStackReader;

mod stream_reader;
pub use stream_reader::BitStreamReader;

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
        println!("vec has {} pairs", test_vec.len());
        for (i, (val, bits)) in test_vec.iter().enumerate().rev() {
            let read_val = dec.read(*bits).unwrap_or_else(|| {
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
