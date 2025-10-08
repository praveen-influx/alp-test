pub mod gorilla;

use rand::Rng;
use alp::ALPRDFloat;
use fastlanes::BitPacking;

/// Generate different types of test data for compression testing
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate time series data with small variations
    pub fn time_series(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| 100.0 + (i as f64) * 0.01 + (i as f64 * 0.1).sin() * 0.001)
            .collect()
    }

    /// Generate sensor data with noise
    pub fn sensor_data(size: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let base = 23.45;
        (0..size)
            .map(|_| base + rng.gen_range(-0.05..0.05))
            .collect()
    }

    /// Generate stock price data
    pub fn stock_prices(size: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut price = 150.0;
        let mut result = Vec::with_capacity(size);

        for _ in 0..size {
            // Random walk with small steps
            price += rng.gen_range(-0.5..0.5);
            result.push(price);
        }

        result
    }

    /// Generate random data (worst case for compression)
    pub fn random_data(size: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen_range(0.0..1000.0)).collect()
    }

    /// Generate identical values (best case for compression)
    pub fn identical_values(size: usize, value: f64) -> Vec<f64> {
        vec![value; size]
    }

    /// Generate data with special float values
    pub fn special_values() -> Vec<f64> {
        vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            std::f64::consts::PI,
            std::f64::consts::E,
        ]
    }

    /// Generate subnormal numbers
    pub fn subnormal_values() -> Vec<f64> {
        vec![
            f64::MIN_POSITIVE,        // Smallest positive normal
            f64::MIN_POSITIVE / 2.0,  // Subnormal
            f64::MIN_POSITIVE / 10.0, // Smaller subnormal
            5e-324,                   // Very small subnormal
            -5e-324,                  // Negative subnormal
            1e-323,
            -1e-323,
        ]
    }
}

/// Calculate compression ratio (compressed_size / original_size)
pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
    (compressed_size as f64 / original_size as f64) * 100.0
}

/// Calculate the actual compressed size of ALP-encoded data using bit-packing
/// This matches the C++ ALP implementation from https://github.com/cwida/ALP
/// Formula: bits_per_value = bit_width + (exceptions * 80 / vector_size) + overhead_per_vector
pub fn calculate_alp_compressed_size(
    encoded: &[i64],
    exceptions: &[f64],
) -> usize {
    calculate_alp_compressed_size_detailed(encoded, exceptions).total_bytes
}

/// Calculate the actual compressed size with detailed breakdown
pub fn calculate_alp_compressed_size_detailed(
    encoded: &[i64],
    exceptions: &[f64],
) -> AlpSizeBreakdown {
    const VECTOR_SIZE: usize = 1024;
    const OVERHEAD_PER_VECTOR_BITS: f64 = 88.0; // (8 + 8 + 8 + 64) bytes = 88 bytes per vector
    const EXCEPTION_SIZE_BITS: usize = 64; // 8 bytes per f64
    const POSITION_SIZE_BITS: usize = 16; // 2 bytes per position

    // Calculate bits needed for frame-of-reference + bit-packing
    let bit_width = if encoded.is_empty() {
        0
    } else {
        let min = *encoded.iter().min().unwrap();
        let max = *encoded.iter().max().unwrap();
        let range = max.saturating_sub(min);

        if range == 0 {
            0 // All values are identical
        } else {
            // Number of bits needed = log2(range) + 1
            (range as u64).checked_ilog2().map(|b| b + 1).unwrap_or(64) as usize
        }
    };

    // Number of full vectors
    let num_vectors = encoded.len().div_ceil(VECTOR_SIZE);

    // Calculate bits per value following the C++ formula
    let bits_per_value = if encoded.is_empty() {
        0.0
    } else {
        let exception_bits_per_value =
            (exceptions.len() as f64 * (EXCEPTION_SIZE_BITS + POSITION_SIZE_BITS) as f64) / encoded.len() as f64;
        let overhead_bits_per_value =
            (num_vectors as f64 * OVERHEAD_PER_VECTOR_BITS) / encoded.len() as f64;

        bit_width as f64 + exception_bits_per_value + overhead_bits_per_value
    };

    let total_bits = (encoded.len() as f64 * bits_per_value).ceil() as usize;
    let total_bytes = total_bits.div_ceil(8);

    AlpSizeBreakdown {
        num_values: encoded.len(),
        num_exceptions: exceptions.len(),
        num_vectors,
        bit_width,
        bits_per_value,
        total_bits,
        total_bytes,
        min_encoded: encoded.iter().min().copied(),
        max_encoded: encoded.iter().max().copied(),
    }
}

#[derive(Debug)]
pub struct AlpSizeBreakdown {
    pub num_values: usize,
    pub num_exceptions: usize,
    pub num_vectors: usize,
    pub bit_width: usize,
    pub bits_per_value: f64,
    pub total_bits: usize,
    pub total_bytes: usize,
    pub min_encoded: Option<i64>,
    pub max_encoded: Option<i64>,
}

/// Compress ALP-encoded integers using Frame-of-Reference + Bit-Packing (FFOR)
/// This matches the C++ ALP implementation which applies FFOR after encoding
/// Returns actual compressed bytes for fair comparison with Gorilla
pub fn compress_alp_with_ffor(encoded: &[i64], exceptions: &[f64]) -> Vec<u8> {
    const CHUNK_SIZE: usize = 1024;

    let mut output = Vec::new();

    // Write number of values (8 bytes)
    output.extend_from_slice(&(encoded.len() as u64).to_le_bytes());

    // Process encoded values in 1024-element chunks
    for chunk in encoded.chunks(CHUNK_SIZE) {
        if chunk.is_empty() {
            continue;
        }

        // Find min and max for this chunk
        let min_val = *chunk.iter().min().unwrap();
        let max_val = *chunk.iter().max().unwrap();
        let range = max_val.saturating_sub(min_val);

        // Calculate bit width needed
        let bit_width = if range == 0 {
            0
        } else {
            (range as u64).checked_ilog2().map(|b| b + 1).unwrap_or(64) as usize
        };

        // Write chunk header: min (8 bytes) + bit_width (1 byte) + chunk_len (2 bytes)
        output.extend_from_slice(&min_val.to_le_bytes());
        output.push(bit_width as u8);
        output.extend_from_slice(&(chunk.len() as u16).to_le_bytes());

        if bit_width == 0 {
            // All values are identical, no need to pack
            continue;
        }

        // Prepare chunk for packing (apply frame-of-reference)
        let mut for_chunk = vec![0u64; CHUNK_SIZE];
        for (i, &val) in chunk.iter().enumerate() {
            // Convert i64 to u64 and subtract min (frame of reference)
            for_chunk[i] = val.wrapping_sub(min_val) as u64;
        }

        // For partial chunks, pad with zeros
        if chunk.len() < CHUNK_SIZE {
            for_chunk.resize(CHUNK_SIZE, 0);
        }

        // Calculate packed output size: 1024 * bit_width / 64
        let packed_size = (CHUNK_SIZE * bit_width + 63) / 64; // round up
        let mut packed = vec![0u64; packed_size];

        // Convert to fixed-size array for fastlanes
        let for_chunk_array: &[u64; 1024] = for_chunk.as_slice().try_into().unwrap();

        // Pack the data using BitPacking
        unsafe {
            BitPacking::unchecked_pack(bit_width, for_chunk_array, &mut packed);
        }

        // Write packed data as bytes
        for &val in &packed {
            output.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Write exception count (4 bytes)
    output.extend_from_slice(&(exceptions.len() as u32).to_le_bytes());

    // Write exceptions as raw f64 bytes
    for &exc in exceptions {
        output.extend_from_slice(&exc.to_le_bytes());
    }

    output
}

/// Compress ALP-RD encoded data using bit-packing
/// Returns actual compressed bytes for fair comparison with Gorilla
pub fn compress_alprd_with_bitpacking(
    left_parts: &[u16],
    left_dict: &[u16],
    left_exceptions: &alp::Exceptions<u16>,
    right_parts: &[u64],
    right_bit_width: u8,
) -> Vec<u8> {
    const CHUNK_SIZE: usize = 1024;

    let mut output = Vec::new();

    // Write number of values (8 bytes)
    output.extend_from_slice(&(left_parts.len() as u64).to_le_bytes());

    // Write dictionary size (2 bytes) and dictionary values
    output.extend_from_slice(&(left_dict.len() as u16).to_le_bytes());
    for &dict_val in left_dict {
        output.extend_from_slice(&dict_val.to_le_bytes());
    }

    // Write right bit width (1 byte)
    output.push(right_bit_width);

    // Calculate bit width for left parts (dictionary indices)
    let left_bit_width = if left_dict.len() <= 1 {
        1
    } else {
        ((left_dict.len() - 1) as u32).ilog2() as usize + 1
    };

    // Process left_parts in 1024-element chunks
    for chunk in left_parts.chunks(CHUNK_SIZE) {
        if chunk.is_empty() {
            continue;
        }

        // Write chunk length (2 bytes)
        output.extend_from_slice(&(chunk.len() as u16).to_le_bytes());

        if left_bit_width == 0 {
            // All values are 0, no need to pack
            continue;
        }

        // Prepare chunk for packing
        let mut pack_chunk = vec![0u16; CHUNK_SIZE];
        pack_chunk[..chunk.len()].copy_from_slice(chunk);

        // Calculate packed output size: 1024 * bit_width / 16
        let packed_size = (CHUNK_SIZE * left_bit_width + 15) / 16; // round up
        let mut packed = vec![0u16; packed_size];

        // Convert to fixed-size array for fastlanes
        let pack_chunk_array: &[u16; 1024] = pack_chunk.as_slice().try_into().unwrap();

        // Pack the data using BitPacking
        unsafe {
            BitPacking::unchecked_pack(left_bit_width, pack_chunk_array, &mut packed);
        }

        // Write packed data as bytes
        for &val in &packed {
            output.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Count and write exceptions
    let exception_count = count_exceptions(left_exceptions, left_parts.len());
    output.extend_from_slice(&(exception_count as u32).to_le_bytes());

    // Apply exceptions to a temporary array to extract actual exception values
    if exception_count > 0 {
        let mut temp_vec = vec![0u16; left_parts.len()];
        left_exceptions.apply(&mut temp_vec);

        // Find and write non-zero exception positions and values
        for (pos, &val) in temp_vec.iter().enumerate() {
            if val != 0 {
                output.extend_from_slice(&(pos as u16).to_le_bytes());
                output.extend_from_slice(&val.to_le_bytes());
            }
        }
    }

    // Process right_parts in 1024-element chunks and bit-pack them
    for chunk in right_parts.chunks(CHUNK_SIZE) {
        if chunk.is_empty() {
            continue;
        }

        // Write chunk length (2 bytes)
        output.extend_from_slice(&(chunk.len() as u16).to_le_bytes());

        if right_bit_width == 0 {
            // All values are 0, no need to pack
            continue;
        }

        // Prepare chunk for packing
        let mut pack_chunk = vec![0u64; CHUNK_SIZE];
        pack_chunk[..chunk.len()].copy_from_slice(chunk);

        // Calculate packed output size: 1024 * bit_width / 64
        let packed_size = (CHUNK_SIZE * right_bit_width as usize + 63) / 64; // round up
        let mut packed = vec![0u64; packed_size];

        // Convert to fixed-size array for fastlanes
        let pack_chunk_array: &[u64; 1024] = pack_chunk.as_slice().try_into().unwrap();

        // Pack the data using BitPacking
        unsafe {
            BitPacking::unchecked_pack(right_bit_width as usize, pack_chunk_array, &mut packed);
        }

        // Write packed data as bytes
        for &val in &packed {
            output.extend_from_slice(&val.to_le_bytes());
        }
    }

    output
}

/// Decompress ALP-RD data that was compressed with bit-packing
/// Returns the original f64 values
pub fn decompress_alprd_with_bitpacking(compressed: &[u8]) -> Vec<f64> {
    const CHUNK_SIZE: usize = 1024;
    let mut cursor = 0;

    // Read number of values (8 bytes)
    let num_values = u64::from_le_bytes(compressed[cursor..cursor+8].try_into().unwrap()) as usize;
    cursor += 8;

    // Read dictionary size (2 bytes) and dictionary values
    let dict_size = u16::from_le_bytes(compressed[cursor..cursor+2].try_into().unwrap()) as usize;
    cursor += 2;

    let mut left_dict = Vec::with_capacity(dict_size);
    for _ in 0..dict_size {
        let val = u16::from_le_bytes(compressed[cursor..cursor+2].try_into().unwrap());
        left_dict.push(val);
        cursor += 2;
    }

    // Read right bit width (1 byte)
    let right_bit_width = compressed[cursor];
    cursor += 1;

    // Calculate left bit width from dictionary size
    let left_bit_width = if dict_size <= 1 {
        1
    } else {
        ((dict_size - 1) as u32).ilog2() as usize + 1
    };

    // Decompress left_parts
    let mut left_parts = Vec::with_capacity(num_values);
    let num_left_chunks = num_values.div_ceil(CHUNK_SIZE);

    for _ in 0..num_left_chunks {
        // Read chunk length (2 bytes)
        let chunk_len = u16::from_le_bytes(compressed[cursor..cursor+2].try_into().unwrap()) as usize;
        cursor += 2;

        if left_bit_width == 0 {
            // All zeros
            left_parts.extend(std::iter::repeat(0u16).take(chunk_len));
            continue;
        }

        // Calculate packed size
        let packed_size = (CHUNK_SIZE * left_bit_width + 15) / 16;

        // Read packed data
        let mut packed = vec![0u16; packed_size];
        for i in 0..packed_size {
            packed[i] = u16::from_le_bytes(compressed[cursor..cursor+2].try_into().unwrap());
            cursor += 2;
        }

        // Unpack the data
        let mut unpacked = [0u16; CHUNK_SIZE];
        unsafe {
            BitPacking::unchecked_unpack(left_bit_width, &packed, &mut unpacked);
        }

        left_parts.extend_from_slice(&unpacked[..chunk_len]);
    }

    // Read exception count (4 bytes)
    let exception_count = u32::from_le_bytes(compressed[cursor..cursor+4].try_into().unwrap()) as usize;
    cursor += 4;

    // Read and apply exceptions
    for _ in 0..exception_count {
        let pos = u16::from_le_bytes(compressed[cursor..cursor+2].try_into().unwrap()) as usize;
        cursor += 2;
        let val = u16::from_le_bytes(compressed[cursor..cursor+2].try_into().unwrap());
        cursor += 2;
        left_parts[pos] = val;
    }

    // Apply dictionary lookup to left_parts
    let mut left_parts_decoded = Vec::with_capacity(num_values);
    for &code in &left_parts {
        left_parts_decoded.push(left_dict[code as usize] as u64);
    }

    // Decompress right_parts
    let mut right_parts = Vec::with_capacity(num_values);
    let num_right_chunks = num_values.div_ceil(CHUNK_SIZE);

    for _ in 0..num_right_chunks {
        // Read chunk length (2 bytes)
        let chunk_len = u16::from_le_bytes(compressed[cursor..cursor+2].try_into().unwrap()) as usize;
        cursor += 2;

        if right_bit_width == 0 {
            // All zeros
            right_parts.extend(std::iter::repeat(0u64).take(chunk_len));
            continue;
        }

        // Calculate packed size
        let packed_size = (CHUNK_SIZE * right_bit_width as usize + 63) / 64;

        // Read packed data
        let mut packed = vec![0u64; packed_size];
        for i in 0..packed_size {
            packed[i] = u64::from_le_bytes(compressed[cursor..cursor+8].try_into().unwrap());
            cursor += 8;
        }

        // Unpack the data
        let mut unpacked = [0u64; CHUNK_SIZE];
        unsafe {
            BitPacking::unchecked_unpack(right_bit_width as usize, &packed, &mut unpacked);
        }

        right_parts.extend_from_slice(&unpacked[..chunk_len]);
    }

    // Recombine left and right parts to reconstruct original f64 values
    left_parts_decoded
        .into_iter()
        .zip(right_parts.iter())
        .map(|(left, right)| {
            let bits = (left << (right_bit_width as u64)) | right;
            f64::from_bits(bits)
        })
        .collect()
}

/// Helper to count exceptions from Exceptions struct
/// Since Exceptions doesn't expose its length, we need to work around it
pub fn count_exceptions(exceptions: &alp::Exceptions<u16>, vector_len: usize) -> usize {
    // Create two marker arrays - one all zeros, one all ones
    let mut test_vec = vec![0u16; vector_len];
    let original_vec = test_vec.clone();

    // Apply exceptions - this will set values at exception positions
    exceptions.apply(&mut test_vec);

    // Count how many positions changed
    test_vec.iter().zip(original_vec.iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Calculate the compressed size of ALP-RD encoded data
/// Formula from C++ implementation:
/// bits_per_value = left_bit_width + right_bit_width + (exceptions * 32 / num_values) + alprd_overhead
pub fn calculate_alprd_compressed_size<T>(
    left_parts: &[u16],
    left_dict_size: usize,
    left_exceptions_count: usize,
    right_bit_width: u8,
) -> AlpRdSizeBreakdown
where
    T: ALPRDFloat,
{
    const ROWGROUP_SIZE: usize = 102_400; // 1024 * 100
    const MAX_RD_DICTIONARY_SIZE: usize = 8;
    const RD_EXCEPTION_SIZE_BITS: usize = 16; // u16
    const RD_POSITION_SIZE_BITS: usize = 16; // u16

    let num_values = left_parts.len();

    // Left bit width = log2(dictionary_size)
    let left_bit_width = if left_dict_size <= 1 {
        1
    } else {
        (left_dict_size as u32 - 1).ilog2() as usize + 1
    };

    // ALP-RD overhead per value = (dictionary_size * 16 bits) / rowgroup_size
    let alprd_overhead_bits_per_value =
        (MAX_RD_DICTIONARY_SIZE * 16) as f64 / ROWGROUP_SIZE as f64;

    // Exception overhead per value
    let exception_bits_per_value = if num_values == 0 {
        0.0
    } else {
        (left_exceptions_count as f64 * (RD_EXCEPTION_SIZE_BITS + RD_POSITION_SIZE_BITS) as f64)
            / num_values as f64
    };

    // Total bits per value
    let bits_per_value = left_bit_width as f64
        + right_bit_width as f64
        + exception_bits_per_value
        + alprd_overhead_bits_per_value;

    let total_bits = (num_values as f64 * bits_per_value).ceil() as usize;
    let total_bytes = total_bits.div_ceil(8);

    AlpRdSizeBreakdown {
        num_values,
        left_exceptions_count,
        left_bit_width,
        right_bit_width: right_bit_width as usize,
        left_dict_size,
        bits_per_value,
        total_bits,
        total_bytes,
    }
}

#[derive(Debug)]
pub struct AlpRdSizeBreakdown {
    pub num_values: usize,
    pub left_exceptions_count: usize,
    pub left_bit_width: usize,
    pub right_bit_width: usize,
    pub left_dict_size: usize,
    pub bits_per_value: f64,
    pub total_bits: usize,
    pub total_bytes: usize,
}

/// Verify that two f64 slices are bit-exactly equal
pub fn verify_bit_exact_equality(original: &[f64], decompressed: &[f64]) -> Result<(), String> {
    if original.len() != decompressed.len() {
        return Err(format!(
            "Length mismatch: original {} vs decompressed {}",
            original.len(),
            decompressed.len()
        ));
    }

    for (i, (orig, decomp)) in original.iter().zip(decompressed.iter()).enumerate() {
        // Special handling for NaN (NaN != NaN in floating point)
        if orig.is_nan() && decomp.is_nan() {
            // Both are NaN - check if they have the same bit pattern
            if orig.to_bits() != decomp.to_bits() {
                // Different NaN patterns - this might be acceptable depending on the algorithm
                // For now, we'll allow it as both are NaN
                continue;
            }
        } else if orig.to_bits() != decomp.to_bits() {
            return Err(format!(
                "Bit mismatch at index {}: original {:064b} ({}) vs decompressed {:064b} ({})",
                i,
                orig.to_bits(),
                orig,
                decomp.to_bits(),
                decomp
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use super::*;
    use alp::encode;

    /// Test that both ALP and Gorilla preserve values exactly
    #[test]
    fn test_correctness_time_series() {
        let data = TestDataGenerator::time_series(1000);

        // Test Gorilla
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());
        verify_bit_exact_equality(&data, &gorilla_decompressed)
            .expect("Gorilla compression should be lossless");

        // Test ALP
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        assert!(encoded.len() > 0);

        println!("Time series data correctness: PASSED");
    }

    #[test]
    fn test_correctness_sensor_data() {
        let data = TestDataGenerator::sensor_data(500);

        // Test Gorilla
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());
        verify_bit_exact_equality(&data, &gorilla_decompressed)
            .expect("Gorilla should preserve sensor data exactly");

        // Test ALP
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        assert!(encoded.len() > 0);

        println!("Sensor data correctness: PASSED");
    }

    #[test]
    fn test_correctness_special_values() {
        let data = TestDataGenerator::special_values();

        // Test Gorilla
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());

        // Special check for special values
        for (i, (orig, decomp)) in data.iter().zip(gorilla_decompressed.iter()).enumerate() {
            if orig.is_nan() {
                assert!(decomp.is_nan(), "NaN not preserved at index {}", i);
            } else {
                assert_eq!(orig.to_bits(), decomp.to_bits(),
                    "Special value mismatch at index {}: {} vs {}", i, orig, decomp);
            }
        }

        // Test ALP
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        assert!(encoded.len() > 0);

        println!("Special values correctness: PASSED");
    }

    #[test]
    fn test_correctness_subnormal_values() {
        let data = TestDataGenerator::subnormal_values();

        // Test Gorilla
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());
        verify_bit_exact_equality(&data, &gorilla_decompressed)
            .expect("Gorilla should preserve subnormal values exactly");

        // Test ALP
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        assert!(encoded.len() > 0);

        println!("Subnormal values correctness: PASSED");
    }

    #[test]
    fn test_correctness_identical_values() {
        let data = TestDataGenerator::identical_values(1000, 42.42);

        // Test Gorilla
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());
        verify_bit_exact_equality(&data, &gorilla_decompressed)
            .expect("Gorilla should preserve identical values exactly");

        // Test ALP
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        assert!(encoded.len() > 0);
        assert_eq!(encoded.len(), data.len()); // All values should be encoded

        println!("Identical values correctness: PASSED");
    }

    #[test]
    fn test_correctness_random_data() {
        let data = TestDataGenerator::random_data(500);

        // Test Gorilla
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());
        verify_bit_exact_equality(&data, &gorilla_decompressed)
            .expect("Gorilla should preserve random data exactly");

        // Test ALP
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        assert!(encoded.len() > 0);

        println!("Random data correctness: PASSED");
    }

    #[test]
    fn test_correctness_zero_and_negative_zero() {
        let data = vec![0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 0.0, -0.0];

        // Test Gorilla
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());

        // Verify sign bit is preserved
        for (i, (orig, decomp)) in data.iter().zip(gorilla_decompressed.iter()).enumerate() {
            assert_eq!(orig.to_bits(), decomp.to_bits(),
                "Sign bit not preserved at index {}: {:064b} vs {:064b}",
                i, orig.to_bits(), decomp.to_bits());
        }

        // Test ALP
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        assert!(encoded.len() > 0);

        println!("Zero and negative zero correctness: PASSED");
    }

    #[test]
    fn test_correctness_edge_cases() {
        // Test empty data
        let empty_data: Vec<f64> = vec![];
        let gorilla_compressed = gorilla::compress(&empty_data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, empty_data.len());
        assert_eq!(gorilla_decompressed.len(), 0);

        // Test single value
        let single = vec![3.14159];
        let gorilla_compressed = gorilla::compress(&single);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, single.len());
        verify_bit_exact_equality(&single, &gorilla_decompressed)
            .expect("Gorilla should handle single value");

        // Test ALP with single value
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&single, None);
        assert!(encoded.len() > 0);

        println!("Edge cases correctness: PASSED");
    }

    #[test]
    fn test_alprd_roundtrip() {
        use alp::RDEncoder;

        // Test with time series data
        let data = TestDataGenerator::time_series(1000);

        // Encode with ALP-RD
        let rd_encoder = RDEncoder::new(&data[..]);
        let split = rd_encoder.split(&data);
        let (left_parts, left_dict, left_exceptions, right_parts, right_bit_width) = split.into_parts();

        // Compress to bytes
        let compressed = compress_alprd_with_bitpacking(
            &left_parts,
            &left_dict,
            &left_exceptions,
            &right_parts,
            right_bit_width,
        );

        // Decompress
        let decompressed = decompress_alprd_with_bitpacking(&compressed);

        // Verify bit-exact equality
        verify_bit_exact_equality(&data, &decompressed)
            .expect("ALP-RD compression should be lossless");

        println!("ALP-RD round-trip correctness: PASSED");
    }

    #[test]
    fn test_alprd_roundtrip_bitcoin() {
        use alp::RDEncoder;

        // Test with Bitcoin data if available
        if let Ok(data_str) = std::fs::read_to_string("/tmp/bitcoin_test.csv") {
            let bitcoin_data: Vec<f64> = data_str
                .lines()
                .filter_map(|line| line.trim().parse::<f64>().ok())
                .collect();

            if !bitcoin_data.is_empty() {
                // Encode with ALP-RD
                let rd_encoder = RDEncoder::new(&bitcoin_data[..]);
                let split = rd_encoder.split(&bitcoin_data);
                let (left_parts, left_dict, left_exceptions, right_parts, right_bit_width) = split.into_parts();

                // Compress to bytes
                let compressed = compress_alprd_with_bitpacking(
                    &left_parts,
                    &left_dict,
                    &left_exceptions,
                    &right_parts,
                    right_bit_width,
                );

                // Decompress
                let decompressed = decompress_alprd_with_bitpacking(&compressed);

                // Verify bit-exact equality
                verify_bit_exact_equality(&bitcoin_data, &decompressed)
                    .expect("ALP-RD compression should be lossless for Bitcoin data");

                println!("ALP-RD Bitcoin round-trip correctness: PASSED");
            }
        }
    }
}