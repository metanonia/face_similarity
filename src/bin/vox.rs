// Copyright (c) 2025 metanonia
//
// This source code is licensed under the MIT License.
// See the LICENSE file in the project root for license terms.
//
// This module implements a VoxCeleb2 face similarity performance test model.
// It randomly samples N IDs from /vox/mp4/, extracts face embeddings from all mp4 videos for each ID,
// compares embeddings pairwise across sampled IDs, and calculates similarity statistics (max, min, mean, median, stddev).
//

use std::path::Path;
use std::fs::{self, File};
use std::io::Write;
use std::collections::HashMap;
use rand::seq::SliceRandom;
use opencv::{videoio, prelude::*, core, imgproc, imgcodecs, Result};
use rand::prelude::IndexedRandom;
use face_similarity::blaze_model::BlazeFaceModel;
use face_similarity::scrfd_model::SCRFDDetector;
use face_similarity::arcface_model::ArcFaceModel;
use face_similarity::face_align::FaceAlign;

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Extract embedding from a frame image Mat using given models
fn extract_embedding_from_frame(
    mat: &Mat,
    blaze: &mut BlazeFaceModel,
    scrfd: &mut SCRFDDetector,
    arcface: &mut ArcFaceModel,
) -> Option<Vec<f32>> {
    if mat.empty() {
        return None;
    }

    let orig_height = mat.rows() as f32;
    let orig_width = mat.cols() as f32;

    // Resize for BlazeFace input
    let mut resized = Mat::default();
    imgproc::resize(mat, &mut resized, core::Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR).ok()?;

    let face_detections = blaze.detect(&resized).ok()?;
    if face_detections.is_empty() {
        return None;
    }

    // Use first detection
    let detection = &face_detections[0];
    let scale_x = orig_width / 128.0;
    let scale_y = orig_height / 128.0;

    let scaled_bbox = opencv::core::Rect::new(
        (detection.bbox.x as f32 * scale_x) as i32,
        (detection.bbox.y as f32 * scale_y) as i32,
        (detection.bbox.width as f32 * scale_x) as i32,
        (detection.bbox.height as f32 * scale_y) as i32,
    );

    // Margin and crop region adjustment
    let margin_ratio = 0.2;
    let margin_x = (scaled_bbox.width as f32 * margin_ratio) as i32;
    let margin_y = (scaled_bbox.height as f32 * margin_ratio) as i32;

    let new_width = scaled_bbox.width + 2 * margin_x;
    let new_height = scaled_bbox.height + 2 * margin_y;
    let max_side = new_width.max(new_height);

    let center_x = scaled_bbox.x + scaled_bbox.width / 2;
    let center_y = scaled_bbox.y + scaled_bbox.height / 2;

    let new_x = (center_x - max_side / 2).max(0);
    let new_y = (center_y - max_side / 2).max(0);
    let new_x = new_x.min(orig_width as i32 - max_side);
    let new_y = new_y.min(orig_height as i32 - max_side);

    let expanded_bbox = opencv::core::Rect::new(
        new_x,
        new_y,
        max_side.min(orig_width as i32 - new_x),
        max_side.min(orig_height as i32 - new_y),
    );

    let mut cropped = Mat::default();
    mat.roi(expanded_bbox).ok()?.copy_to(&mut cropped).ok()?;

    // Detect landmarks with SCRFD
    let landmark_detects = scrfd.detect(mat).ok()?;
    if landmark_detects.is_empty() {
        return None;
    }

    let landmark = &landmark_detects[0];

    // Face align and embedding extraction
    let aligned = FaceAlign::norm_crop(mat, &landmark.landmarks, 112).ok()?;
    let safe_bbox = opencv::core::Rect::new(0, 0, 112, 112);
    let embedding = arcface.embbeding(&aligned, safe_bbox).ok()?;

    Some(embedding)
}

/// Extract embeddings from video frames every 5 frames, up to all frames
fn extract_embeddings_from_video(
    video_path: &str,
    blaze: &mut BlazeFaceModel,
    scrfd: &mut SCRFDDetector,
    arcface: &mut ArcFaceModel,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut cap = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(format!("Can't open video file {}", video_path).into());
    }

    let mut embeddings = Vec::new();
    let mut frame_idx = 0;
    let mut mat = Mat::default();

    while cap.read(&mut mat)? {
        if frame_idx % 5 == 0 {
            if let Some(embedding) = extract_embedding_from_frame(&mat, blaze, scrfd, arcface) {
                embeddings.push(embedding);
            }
        }
        frame_idx += 1;
    }
    Ok(embeddings)
}

/// Compute average embedding vector from multiple embeddings
fn average_embedding(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    if vectors.is_empty() {
        return None;
    }
    let length = vectors[0].len();
    let mut avg = vec![0.0; length];
    let count = vectors.len() as f32;

    for vec in vectors {
        for i in 0..length {
            avg[i] += vec[i];
        }
    }
    for i in 0..length {
        avg[i] /= count;
    }
    Some(avg)
}

/// Compute max, min, mean, median, stddev of similarities
fn compute_statistics(similarities: &[f32]) -> (f32, f32, f32, f32, f32) {
    let max = similarities.iter().cloned().fold(f32::MIN, f32::max);
    let min = similarities.iter().cloned().fold(f32::MAX, f32::min);
    let mean = similarities.iter().sum::<f32>() / similarities.len() as f32;

    let mut sorted = similarities.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    let variance = similarities.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / similarities.len() as f32;
    let stddev = variance.sqrt();

    (max, min, mean, median, stddev)
}

/// List all mp4 files under an ID directory with intermediate subfolders
fn list_mp4_files_recursive(id_dir: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut mp4_files = Vec::new();
    for entry in fs::read_dir(id_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // 하위 폴더 내 재귀적으로 mp4 검색
            let inner_files = list_mp4_files_recursive(path.to_str().unwrap())?;
            mp4_files.extend(inner_files);
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "mp4" {
                    mp4_files.push(path.to_string_lossy().into_owned());
                }
            }
        }
    }
    Ok(mp4_files)
}

/// List all available IDs (folder names) under /vox/mp4/
fn list_all_ids(base_dir: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut ids = Vec::new();
    for entry in fs::read_dir(base_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            ids.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    Ok(ids)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <number_of_ids_to_sample>", args[0]);
        return Ok(());
    }

    let sample_count = args[1].parse::<usize>().unwrap_or(1);
    let base_dir = "vox/test_mp4/";

    // Initialize models
    let mut blaze = BlazeFaceModel::new("models/blaze.onnx", 0.5, 0.3, 2)?;
    let mut scrfd = SCRFDDetector::new("models/det_500m.onnx", 0.5, 0.25, 320)?;
    let mut arcface = ArcFaceModel::new("models/w600k_mbf.onnx")?;

    // 1. List all IDs and randomly select N
    let all_ids = list_all_ids(base_dir)?;
    if sample_count > all_ids.len() {
        eprintln!("Requested sample count {} exceeds available IDs {}", sample_count, all_ids.len());
        return Ok(());
    }

    let mut rng = rand::thread_rng();
    let sampled_ids: Vec<String> = all_ids.choose_multiple(&mut rng, sample_count).cloned().collect();
    println!("Sampled IDs: {:?}", &sampled_ids);

    // 2. Extract average embeddings for all videos per ID
    let mut id_video_embeddings: HashMap<String, Vec<(String, Vec<f32>)>> = HashMap::new();

    for id in &sampled_ids {
        let id_dir = format!("{}{}", base_dir, id);
        let video_files = list_mp4_files_recursive(&id_dir)?;
        println!("ID {} has {} videos", id, video_files.len());

        let mut videos_embeddings = Vec::new();
        for video_path in video_files {
            println!("Processing video: {}", video_path);
            let embeddings = extract_embeddings_from_video(&video_path, &mut blaze, &mut scrfd, &mut arcface)?;
            if embeddings.len() < 5 {
                eprintln!("Warning: less than 5 embeddings extracted, skipping average for {}", video_path);
                continue;
            }
            // Average first 5 embeddings (25 frames)
            let avg_embedding = average_embedding(&embeddings[..5]).unwrap();
            videos_embeddings.push((video_path, avg_embedding));
        }
        id_video_embeddings.insert(id.clone(), videos_embeddings);
    }

    // 3. Cross compare embeddings pairwise across sampled IDs
    println!("Computing pairwise similarity statistics...");
    let mut results = Vec::new();

    // For each ID pair (including same ID)
    for i in 0..sampled_ids.len() {
        let id_a = &sampled_ids[i];
        for j in i..sampled_ids.len() {
            let id_b = &sampled_ids[j];

            let videos_a = &id_video_embeddings[id_a];
            let videos_b = &id_video_embeddings[id_b];

            for (path_a, emb_a) in videos_a.iter() {
                for (path_b, emb_b) in videos_b.iter() {
                    let similarity = cosine_similarity(emb_a, emb_b);
                    results.push((
                        id_a.clone(), path_a.clone(),
                        id_b.clone(), path_b.clone(),
                        similarity
                    ));
                }
            }
        }
    }

    // 4. Aggregate statistics per ID pair
    use std::collections::BTreeMap;
    let mut stats_map: BTreeMap<(String, String), Vec<f32>> = BTreeMap::new();
    for (id_a, _, id_b, _, sim) in &results {
        let key = if id_a <= id_b { (id_a.clone(), id_b.clone()) } else { (id_b.clone(), id_a.clone()) };
        stats_map.entry(key).or_insert_with(Vec::new).push(*sim);
    }

    // 5. Output statistics CSV
    let mut output = File::create("voxceleb2_similarity_stats.csv")?;
    writeln!(output, "ID_A,ID_B,Max,Min,Mean,Median,StdDev")?;

    for ((id_a, id_b), sims) in &stats_map {
        let (max, min, mean, median, stddev) = compute_statistics(sims);
        writeln!(output, "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}", id_a, id_b, max, min, mean, median, stddev)?;
        println!("{} vs {} => max: {:.4}, min: {:.4}, mean: {:.4}, median: {:.4}, stddev: {:.4}", id_a, id_b, max, min, mean, median, stddev);
    }

    println!("Similarity statistics saved to voxceleb2_similarity_stats.csv");

    Ok(())
}
