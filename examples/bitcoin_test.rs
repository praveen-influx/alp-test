use alp::{encode};
use alp_test::calculate_alp_compressed_size_detailed;
use std::fs;

fn main() {
    // Load Bitcoin price data
    let data_str = fs::read_to_string("/tmp/bitcoin_test.csv").expect("Failed to read file");
    let data: Vec<f64> = data_str
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect();

    println!("Bitcoin Price Data Test");
    println!("=======================");
    println!("Number of values: {}", data.len());
    println!("Original size: {} bytes ({} KB)", data.len() * 8, (data.len() * 8) / 1024);

    // Show first few values
    println!("\nFirst 10 values:");
    for (i, &val) in data.iter().take(10).enumerate() {
        println!("  [{}] = {}", i, val);
    }

    // Encode with ALP
    let (exponents, encoded, _exceptions_pos, exceptions) = encode(&data, None);

    // Calculate compressed size with detailed breakdown
    let breakdown = calculate_alp_compressed_size_detailed(&encoded, &exceptions);

    println!("\n\nALP Compression Results:");
    println!("========================");
    println!("Exponents: e={}, f={}", exponents.e, exponents.f);
    println!("Encoded values: {}", breakdown.num_values);
    println!("Exceptions: {}", breakdown.num_exceptions);
    println!("Num vectors: {}", breakdown.num_vectors);
    println!();
    println!("Encoded integer range:");
    println!("  Min: {:?}", breakdown.min_encoded);
    println!("  Max: {:?}", breakdown.max_encoded);
    if let (Some(min), Some(max)) = (breakdown.min_encoded, breakdown.max_encoded) {
        println!("  Range: {}", max - min);
    }
    println!();
    println!("Bit-packing:");
    println!("  Bit width: {} bits", breakdown.bit_width);
    println!("  Bits per value: {:.2} bits", breakdown.bits_per_value);
    println!();
    println!("Compressed size:");
    println!("  Total bits: {} bits", breakdown.total_bits);
    println!("  Total bytes: {} bytes ({:.2} KB)", breakdown.total_bytes, breakdown.total_bytes as f64 / 1024.0);
    println!();
    println!("Compression ratio: {:.2}%",
        (breakdown.total_bytes as f64 / (data.len() * 8) as f64) * 100.0);
    println!("Space saved: {:.2}%",
        (1.0 - (breakdown.total_bytes as f64 / (data.len() * 8) as f64)) * 100.0);
}
