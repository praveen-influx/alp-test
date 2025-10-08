//! Gorilla compression implementation for f64 values
//! Based on Facebook's Gorilla TSDB paper

use bytes::{BufMut, BytesMut};

/// A bit writer that accumulates bits and writes complete bytes to a buffer.
pub struct BitWriter<'a> {
    buffer: &'a mut BytesMut,
    current_byte: u8,
    bits_in_current_byte: u8,
}

impl<'a> BitWriter<'a> {
    pub fn new(buffer: &'a mut BytesMut) -> Self {
        Self {
            buffer,
            current_byte: 0,
            bits_in_current_byte: 0,
        }
    }

    pub fn write_bit(&mut self, bit: bool) -> Result<(), String> {
        if bit {
            self.current_byte |= 1 << (7 - self.bits_in_current_byte);
        }
        self.bits_in_current_byte += 1;

        if self.bits_in_current_byte == 8 {
            self.buffer.put_u8(self.current_byte);
            self.current_byte = 0;
            self.bits_in_current_byte = 0;
        }
        Ok(())
    }

    pub fn write_bits(&mut self, value: u64, num_bits: usize) -> Result<(), String> {
        if num_bits > 64 {
            return Err(format!("Cannot write {} bits (max 64)", num_bits));
        }

        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1 == 1;
            self.write_bit(bit)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if self.bits_in_current_byte > 0 {
            self.buffer.put_u8(self.current_byte);
            self.current_byte = 0;
            self.bits_in_current_byte = 0;
        }
        Ok(())
    }
}

/// A bit reader that reads bits from a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_offset: usize,
    bit_offset: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_offset: 0,
            bit_offset: 0,
        }
    }

    pub fn read_bit(&mut self) -> Result<bool, String> {
        if self.byte_offset >= self.data.len() {
            return Err("Bit reader reached end of data".into());
        }

        let byte = self.data[self.byte_offset];
        let bit = (byte >> (7 - self.bit_offset)) & 1 == 1;

        self.bit_offset += 1;
        if self.bit_offset == 8 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }

        Ok(bit)
    }

    pub fn read_bits(&mut self, num_bits: usize) -> Result<u64, String> {
        if num_bits > 64 {
            return Err(format!("Cannot read {} bits (max 64)", num_bits));
        }

        let mut value = 0u64;
        for _ in 0..num_bits {
            value = (value << 1) | (self.read_bit()? as u64);
        }
        Ok(value)
    }
}

/// Compress a slice of f64 values using Gorilla compression
pub fn compress(values: &[f64]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut buffer = BytesMut::new();
    let mut bit_writer = BitWriter::new(&mut buffer);

    let mut prev_value = 0u64;
    let mut prev_leading_zeros = 0u8;
    let mut prev_trailing_zeros = 0u8;

    for (i, &value) in values.iter().enumerate() {
        let value_bits = value.to_bits();

        if i == 0 {
            // First value stored as-is
            bit_writer.write_bits(value_bits, 64).unwrap();
            prev_value = value_bits;
        } else {
            let xor = value_bits ^ prev_value;

            if xor == 0 {
                // Values are the same
                bit_writer.write_bit(false).unwrap();
            } else {
                // Values differ
                bit_writer.write_bit(true).unwrap();

                let leading_zeros = xor.leading_zeros() as u8;
                let trailing_zeros = xor.trailing_zeros() as u8;

                // Check if we can reuse the previous bit range
                if prev_leading_zeros != 0
                    && leading_zeros >= prev_leading_zeros
                    && trailing_zeros >= prev_trailing_zeros
                {
                    // Can reuse previous bit range
                    bit_writer.write_bit(false).unwrap();
                    let bits_to_write = 64 - prev_leading_zeros - prev_trailing_zeros;
                    let shifted_xor = xor >> prev_trailing_zeros;
                    bit_writer.write_bits(shifted_xor, bits_to_write as usize).unwrap();
                } else {
                    // Need new bit range
                    bit_writer.write_bit(true).unwrap();

                    // Write 6-bit leading zeros count
                    bit_writer.write_bits(leading_zeros as u64, 6).unwrap();

                    // Calculate and write 6-bit length of meaningful bits
                    let meaningful_bits = 64 - leading_zeros - trailing_zeros;
                    let encoded_length = meaningful_bits - 1; // store length - 1 per Gorilla spec
                    bit_writer.write_bits(encoded_length as u64, 6).unwrap();

                    // Write the meaningful bits
                    let shifted_xor = xor >> trailing_zeros;
                    bit_writer.write_bits(shifted_xor, meaningful_bits as usize).unwrap();

                    prev_leading_zeros = leading_zeros;
                    prev_trailing_zeros = trailing_zeros;
                }
            }
            prev_value = value_bits;
        }
    }

    bit_writer.flush().unwrap();
    buffer.to_vec()
}

/// Decompress Gorilla-compressed data back to f64 values
pub fn decompress(data: &[u8], count: usize) -> Vec<f64> {
    if count == 0 || data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(count);
    let mut bit_reader = BitReader::new(data);

    let mut prev_value = 0u64;
    let mut prev_leading_zeros = 0u8;
    let mut prev_trailing_zeros = 0u8;

    for i in 0..count {
        if i == 0 {
            // First value is stored as-is
            prev_value = bit_reader.read_bits(64).unwrap();
            result.push(f64::from_bits(prev_value));
        } else {
            // Read control bit
            if !bit_reader.read_bit().unwrap() {
                // Value is the same as previous
                result.push(f64::from_bits(prev_value));
            } else {
                // Value differs from previous
                let control_bit = bit_reader.read_bit().unwrap();

                let xor = if !control_bit {
                    // Reuse previous bit range
                    if prev_leading_zeros == 0 && prev_trailing_zeros == 0 {
                        panic!("Cannot reuse bit range when none was established");
                    }
                    let bits_to_read = 64 - prev_leading_zeros - prev_trailing_zeros;
                    if bits_to_read == 0 || prev_trailing_zeros >= 64 {
                        0u64
                    } else {
                        let meaningful_bits = bit_reader.read_bits(bits_to_read as usize).unwrap();
                        meaningful_bits << prev_trailing_zeros
                    }
                } else {
                    // New bit range
                    let leading_zeros = bit_reader.read_bits(6).unwrap() as u8;
                    let encoded_length = bit_reader.read_bits(6).unwrap() as u8;
                    let meaningful_bits_count = encoded_length + 1;

                    if leading_zeros + meaningful_bits_count > 64 {
                        panic!("Invalid bit range: {} leading + {} meaningful > 64",
                               leading_zeros, meaningful_bits_count);
                    }

                    let trailing_zeros = 64 - leading_zeros - meaningful_bits_count;
                    let meaningful_bits = bit_reader.read_bits(meaningful_bits_count as usize).unwrap();

                    if trailing_zeros >= 64 {
                        panic!("Invalid trailing_zeros: {}", trailing_zeros);
                    }

                    prev_leading_zeros = leading_zeros;
                    prev_trailing_zeros = trailing_zeros;

                    meaningful_bits << trailing_zeros
                };

                prev_value ^= xor;
                result.push(f64::from_bits(prev_value));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gorilla_basic() {
        let values = vec![1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1];
        let compressed = compress(&values);
        let decompressed = decompress(&compressed, values.len());

        assert_eq!(values.len(), decompressed.len());
        for (original, decompressed) in values.iter().zip(decompressed.iter()) {
            assert_eq!(original.to_bits(), decompressed.to_bits());
        }
    }

    #[test]
    fn test_gorilla_special_values() {
        let values = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN,
            f64::MAX,
            f64::EPSILON,
        ];

        let compressed = compress(&values);
        let decompressed = decompress(&compressed, values.len());

        assert_eq!(values.len(), decompressed.len());
        for (original, decompressed) in values.iter().zip(decompressed.iter()) {
            if original.is_nan() {
                assert!(decompressed.is_nan());
            } else {
                assert_eq!(original.to_bits(), decompressed.to_bits());
            }
        }
    }

    #[test]
    fn test_gorilla_identical_values() {
        let values = vec![42.42; 100];
        let compressed = compress(&values);
        let decompressed = decompress(&compressed, values.len());

        assert_eq!(values, decompressed);

        // Should be very small for identical values
        assert!(compressed.len() < values.len() * 2);
    }

    #[test]
    fn test_gorilla_time_series() {
        let values: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64) * 0.01 + (i as f64).sin() * 0.001)
            .collect();

        let compressed = compress(&values);
        let decompressed = decompress(&compressed, values.len());

        assert_eq!(values.len(), decompressed.len());
        for (original, decompressed) in values.iter().zip(decompressed.iter()) {
            assert_eq!(original.to_bits(), decompressed.to_bits());
        }
    }
}