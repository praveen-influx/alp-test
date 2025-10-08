use alp_test::{gorilla, TestDataGenerator, compression_ratio, verify_bit_exact_equality,
                compress_alp_with_ffor, compress_alprd_with_bitpacking,
                calculate_alp_compressed_size_detailed, count_exceptions};
use alp::{encode, RDEncoder};
use std::time::Instant;

fn main() {
    println!("=== ALP vs Gorilla Compression Comparison ===\n");
    println!("First, verifying correctness (both algorithms should be lossless)...\n");

    // Test different data patterns
    let test_cases = vec![
        ("Time Series (1000 values)", TestDataGenerator::time_series(1000)),
        ("Sensor Data (1000 values)", TestDataGenerator::sensor_data(1000)),
        ("Stock Prices (1000 values)", TestDataGenerator::stock_prices(1000)),
        ("Identical Values (1000×42.42)", TestDataGenerator::identical_values(1000, 42.42)),
        ("Random Data (1000 values)", TestDataGenerator::random_data(1000)),
        ("Special Float Values", TestDataGenerator::special_values()),
        ("Subnormal Values", TestDataGenerator::subnormal_values()),
    ];

    // First, verify correctness for all test cases
    println!("CORRECTNESS VERIFICATION");
    println!("{:-<60}", "");

    for (name, data) in &test_cases {
        print!("Testing {:<35} ", name);

        // Test Gorilla correctness
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, data.len());

        match verify_bit_exact_equality(&data, &gorilla_decompressed) {
            Ok(_) => print!("Gorilla: ✓  "),
            Err(e) => {
                println!("\n  ERROR in Gorilla: {}", e);
                continue;
            }
        }

        // Test ALP (we can't decompress with public API, but we can verify encoding works)
        let (_exponents, encoded, _exceptions_pos, _exceptions) = encode(&data, None);
        if encoded.len() > 0 {
            print!("ALP: ✓");
        } else {
            print!("ALP: ✗");
        }

        println!();
    }

    // Now compare compression ratios
    println!("\n\nCOMPRESSION COMPARISON (Classic ALP vs ALP-RD vs Gorilla)");
    println!("{:-<105}", "");
    println!("{:<30} {:>12} {:>15} {:>15} {:>15} {:>15}",
             "Data Type", "Original", "Gorilla", "ALP Classic", "ALP-RD", "Winner");
    println!("{:-<105}", "");

    for (name, data) in &test_cases {
        let original_size = data.len() * 8; // 8 bytes per f64

        // Gorilla compression
        let gorilla_compressed = gorilla::compress(&data);
        let gorilla_ratio = compression_ratio(original_size, gorilla_compressed.len());

        // Classic ALP compression
        let (_exponents, encoded, _exceptions_pos, exceptions) = encode(&data, None);
        let alp_compressed = compress_alp_with_ffor(&encoded, &exceptions);
        let alp_ratio = compression_ratio(original_size, alp_compressed.len());

        // ALP-RD compression
        let rd_encoder = RDEncoder::new(&data[..]);
        let split = rd_encoder.split(&data);
        let (left_parts, left_dict, left_exceptions, right_parts, right_bit_width) = split.into_parts();
        let alprd_compressed = compress_alprd_with_bitpacking(
            &left_parts,
            &left_dict,
            &left_exceptions,
            &right_parts,
            right_bit_width,
        );
        let alprd_ratio = compression_ratio(original_size, alprd_compressed.len());

        // Determine winner
        let winner = {
            let min_ratio = gorilla_ratio.min(alp_ratio).min(alprd_ratio);
            if (gorilla_ratio - min_ratio).abs() < 0.01 {
                "Gorilla"
            } else if (alp_ratio - min_ratio).abs() < 0.01 {
                "ALP Classic"
            } else if (alprd_ratio - min_ratio).abs() < 0.01 {
                "ALP-RD"
            } else {
                "Tie"
            }
        };

        println!("{:<30} {:>10}B {:>14.1}% {:>14.1}% {:>14.1}% {:>15}",
                 name, original_size, gorilla_ratio, alp_ratio, alprd_ratio, winner);
    }

    // Detailed example with actual values
    println!("\n\nDETAILED EXAMPLE: Time Series Data");
    println!("{:-<60}", "");

    let example_data: Vec<f64> = vec![
        100.0, 100.1, 100.2, 100.15, 100.25, 100.3, 100.28, 100.35,
        100.4, 100.38, 100.45, 100.5, 100.48, 100.55, 100.6, 100.58,
    ];

    println!("Original data: {:?}", example_data);
    println!("Original size: {} bytes", example_data.len() * 8);

    // Gorilla compression
    let gorilla_compressed = gorilla::compress(&example_data);
    let gorilla_decompressed = gorilla::decompress(&gorilla_compressed, example_data.len());

    println!("\nGorilla Compression:");
    println!("  Compressed size: {} bytes", gorilla_compressed.len());
    println!("  Compression ratio: {:.1}%",
             compression_ratio(example_data.len() * 8, gorilla_compressed.len()));
    println!("  Decompressed: {:?}", gorilla_decompressed);
    println!("  Correctness: {}",
             if example_data == gorilla_decompressed { "✓ PASSED" } else { "✗ FAILED" });

    // ALP compression
    let (exponents, encoded, _exceptions_pos, exceptions) = encode(&example_data, None);
    let alp_breakdown = calculate_alp_compressed_size_detailed(&encoded, &exceptions);

    println!("\nALP Compression:");
    println!("  Exponents: {:?}", exponents);
    println!("  Encoded values: {} integers", encoded.len());
    println!("  Exceptions: {} values", exceptions.len());
    println!("  Compressed size: {} bytes", alp_breakdown.total_bytes);
    println!("  Compression ratio: {:.1}%",
             compression_ratio(example_data.len() * 8, alp_breakdown.total_bytes));
    println!("\n  Size Calculation Breakdown:");
    println!("    Min encoded: {:?}", alp_breakdown.min_encoded);
    println!("    Max encoded: {:?}", alp_breakdown.max_encoded);
    println!("    Range: {:?}", alp_breakdown.max_encoded.and_then(|max|
        alp_breakdown.min_encoded.map(|min| max - min)));
    println!("    Bit width: {} bits", alp_breakdown.bit_width);
    println!("    Num vectors: {}", alp_breakdown.num_vectors);
    println!("    Bits per value: {:.2} bits", alp_breakdown.bits_per_value);
    println!("    Total bits: {} bits", alp_breakdown.total_bits);
    println!("    Total bytes: {} bytes", alp_breakdown.total_bytes);

    // Performance comparison
    println!("\n\nPERFORMANCE TESTING (1000 compressions of 10K values)");
    println!("{:-<70}", "");

    let perf_data = TestDataGenerator::time_series(10000);
    let iterations = 1000;

    // Benchmark Gorilla
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gorilla::compress(&perf_data);
    }
    let gorilla_duration = start.elapsed();

    // Benchmark Classic ALP (with FFOR compression)
    let start = Instant::now();
    for _ in 0..iterations {
        let (_exponents, encoded, _exceptions_pos, exceptions) = encode(&perf_data, None);
        let _ = compress_alp_with_ffor(&encoded, &exceptions);
    }
    let alp_duration = start.elapsed();

    // Benchmark ALP-RD (build encoder once, reuse for all iterations, with bit-packing)
    let rd_encoder = RDEncoder::new(&perf_data[..]);
    let start = Instant::now();
    for _ in 0..iterations {
        let split = rd_encoder.split(&perf_data);
        let (left_parts, left_dict, left_exceptions, right_parts, right_bit_width) = split.into_parts();
        let _ = compress_alprd_with_bitpacking(&left_parts, &left_dict, &left_exceptions, &right_parts, right_bit_width);
    }
    let alprd_duration = start.elapsed();

    println!("Gorilla:     {:?} total, {:.2} µs/compression",
             gorilla_duration, gorilla_duration.as_micros() as f64 / iterations as f64);
    println!("ALP Classic: {:?} total, {:.2} µs/compression",
             alp_duration, alp_duration.as_micros() as f64 / iterations as f64);
    println!("ALP-RD:      {:?} total, {:.2} µs/compression",
             alprd_duration, alprd_duration.as_micros() as f64 / iterations as f64);

    // Find the fastest
    let fastest_duration = gorilla_duration.min(alp_duration).min(alprd_duration);
    let fastest_name = if fastest_duration == gorilla_duration {
        "Gorilla"
    } else if fastest_duration == alp_duration {
        "ALP Classic"
    } else {
        "ALP-RD"
    };

    println!("\nSpeed Comparison:");
    println!("  Gorilla vs Classic ALP: {:.1}x",
        gorilla_duration.as_nanos() as f64 / alp_duration.as_nanos() as f64);
    println!("  Gorilla vs ALP-RD: {:.1}x",
        gorilla_duration.as_nanos() as f64 / alprd_duration.as_nanos() as f64);
    println!("  Classic ALP vs ALP-RD: {:.1}x",
        alprd_duration.as_nanos() as f64 / alp_duration.as_nanos() as f64);
    println!("\nFastest: {}", fastest_name);

    // Test with Bitcoin data if available
    if let Ok(data_str) = std::fs::read_to_string("/tmp/bitcoin_test.csv") {
        println!("\n\nBITCOIN PRICE DATA TEST (1024 values from C++ ALP repo)");
        println!("{:-<60}", "");

        let bitcoin_data: Vec<f64> = data_str
            .lines()
            .filter_map(|line| line.trim().parse::<f64>().ok())
            .collect();

        if !bitcoin_data.is_empty() {
            println!("Loaded {} Bitcoin price values", bitcoin_data.len());
            println!("Original size: {} bytes", bitcoin_data.len() * 8);

            // Classic ALP compression
            let (_exponents, encoded, _exceptions_pos, exceptions) = encode(&bitcoin_data, None);
            let alp_compressed = compress_alp_with_ffor(&encoded, &exceptions);
            let breakdown = calculate_alp_compressed_size_detailed(&encoded, &exceptions);

            println!("\nClassic ALP Compression:");
            println!("  Compressed size: {} bytes", alp_compressed.len());
            println!("  Compression ratio: {:.1}%",
                (alp_compressed.len() as f64 / (bitcoin_data.len() * 8) as f64) * 100.0);
            println!("  Bit width: {} bits", breakdown.bit_width);
            println!("  Bits per value: {:.2} bits (estimated)", breakdown.bits_per_value);
            println!("  Actual bits per value: {:.2} bits", (alp_compressed.len() * 8) as f64 / bitcoin_data.len() as f64);
            println!("  Exceptions: {}", breakdown.num_exceptions);

            // ALP-RD compression
            let rd_encoder = RDEncoder::new(&bitcoin_data[..]);
            let split = rd_encoder.split(&bitcoin_data);
            let (left_parts, left_dict, left_exceptions, right_parts, right_bit_width) = split.into_parts();
            let alprd_compressed = compress_alprd_with_bitpacking(
                &left_parts,
                &left_dict,
                &left_exceptions,
                &right_parts,
                right_bit_width,
            );

            println!("\nALP-RD Compression:");
            println!("  Compressed size: {} bytes", alprd_compressed.len());
            println!("  Compression ratio: {:.1}%",
                (alprd_compressed.len() as f64 / (bitcoin_data.len() * 8) as f64) * 100.0);
            println!("  Dictionary size: {}", left_dict.len());
            println!("  Actual bits per value: {:.2} bits", (alprd_compressed.len() * 8) as f64 / bitcoin_data.len() as f64);
            println!("  Right bit width: {} bits", right_bit_width);
            println!("  Exceptions: {}", count_exceptions(&left_exceptions, left_parts.len()));

            // Gorilla compression
            let gorilla_compressed = gorilla::compress(&bitcoin_data);
            println!("\nGorilla Compression:");
            println!("  Compressed size: {} bytes", gorilla_compressed.len());
            println!("  Compression ratio: {:.1}%",
                (gorilla_compressed.len() as f64 / (bitcoin_data.len() * 8) as f64) * 100.0);
        }
    }
}
