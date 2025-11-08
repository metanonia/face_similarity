// Copyright (c) 2025 metanonia
//
// This source code is licensed under the MIT License.
// See the LICENSE file in the project root for license terms.
//
// This module implements a face similarity performance test model
// using the Labeled Faces in the Wild (LFW) dataset samples.
// The model calculates embeddings and compares facial similarity scores

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::collections::HashMap;
use std::path::Path;
use opencv::{imgcodecs, imgproc, Result};
use opencv::core::{Mat, Size, Vector};
use opencv::prelude::MatTraitConst;

use face_similarity::blaze_model::BlazeFaceModel;
use face_similarity::scrfd_model::SCRFDDetector;
use face_similarity::arcface_model::ArcFaceModel;
use face_similarity::face_align::FaceAlign;

/// 코사인 유사도 계산
fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
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

/// 이미지에서 임베딩 추출
fn extract_embedding(
    image_path: &str,
    blaze: &mut BlazeFaceModel,
    scrfd: &mut SCRFDDetector,
    arcface: &mut ArcFaceModel,
) -> Option<Vec<f32>> {
    // 이미지 읽기
    let src = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).ok()?;
    if src.empty() {
        return None;
    }

    let orig_height = src.rows() as f32;
    let orig_width = src.cols() as f32;

    // Blaze 감지 (얼굴 대략적 위치)
    let mut resized = Mat::default();
    imgproc::resize(&src, &mut resized, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR).ok()?;

    let face_detections = blaze.detect(&resized).ok()?;
    if face_detections.is_empty() {
        eprintln!("No face_detections found");
        return None;
    }

    // 첫 번째 얼굴만 처리
    let detection = &face_detections[0];
    let scale_x = orig_width / 128.0;
    let scale_y = orig_height / 128.0;

    let scaled_bbox = opencv::core::Rect::new(
        (detection.bbox.x as f32 * scale_x) as i32,
        (detection.bbox.y as f32 * scale_y) as i32,
        (detection.bbox.width as f32 * scale_x) as i32,
        (detection.bbox.height as f32 * scale_y) as i32,
    );

    // 바운딩 박스 확장
    let margin_ratio = 0.2;
    let margin_x = (scaled_bbox.width as f32 * margin_ratio) as i32;
    let margin_y = (scaled_bbox.height as f32 * margin_ratio) as i32;

    // 우선 여유를 포함한 확장된 크기 계산
    let new_width = scaled_bbox.width + 2 * margin_x;
    let new_height = scaled_bbox.height + 2 * margin_y;

    // width, height 중 큰 값을 사용해 정사각형으로 만들기
    let max_side = new_width.max(new_height);

    // 정사각형이 중심을 유지하도록 x, y 조정
    let center_x = scaled_bbox.x + scaled_bbox.width / 2;
    let center_y = scaled_bbox.y + scaled_bbox.height / 2;

    let new_x = (center_x - max_side / 2).max(0);
    let new_y = (center_y - max_side / 2).max(0);

    // 이미지 범위를 벗어나지 않게 조정
    let new_x = new_x.min(orig_width as i32 - max_side);
    let new_y = new_y.min(orig_height as i32 - max_side);

    let expanded_bbox = opencv::core::Rect::new(
        new_x,
        new_y,
        max_side.min(orig_width as i32 - new_x),
        max_side.min(orig_height as i32 - new_y),
    );

    // 크롭
    let mut cropped = Mat::default();
    src.roi(expanded_bbox).ok()?.copy_to(&mut cropped).ok()?;

    // SCRFD 랜드마크 감지
    let landmark_detects = scrfd.detect(&cropped).ok()?;
    // println!("  🎯 SCRFD 감지: {}개", landmark_detects.len());
    if landmark_detects.is_empty() {
        eprintln!("No landmarks found");
        return None;
    }

    let landmark = &landmark_detects[0];

    // 정렬 및 임베딩 추출
    let aligned = FaceAlign::norm_crop(&cropped, &landmark.landmarks, 112).ok()?;
    let safe_bbox = opencv::core::Rect::new(0, 0, 112, 112);
    let embedding = arcface.embbeding(&aligned, safe_bbox).ok()?;
    Some(embedding)
}

/// CSV ID를 4자리 포맷으로 변환하고 파일 찾기
fn find_image_file_with_padding(dir_path: &str, image_id: &str, person_name: &str) -> Option<String> {
    let id_trimmed = image_id.trim();

    if !Path::new(dir_path).exists() {
        return None;
    }

    // 숫자를 4자리로 패딩
    if let Ok(id_num) = id_trimmed.parse::<u32>() {
        let padded_id = format!("{:04}", id_num);

        // 시도할 파일명들
        let filenames = vec![
            format!("{}_{}.jpg", person_name, padded_id),     // Ahmed_Chalabi_0002.jpg
            format!("{}_{}.JPG", person_name, padded_id),     // Ahmed_Chalabi_0002.JPG
            format!("{}.jpg", padded_id),                     // 0002.jpg
            format!("{}.JPG", padded_id),                     // 0002.JPG
            format!("{}.jpg", id_trimmed),                    // 2.jpg
            format!("{}.JPG", id_trimmed),                    // 2.JPG
        ];

        for filename in filenames {
            let full_path = format!("{}/{}", dir_path, filename);
            if Path::new(&full_path).exists() {
                return Some(full_path);
            }
        }
    }

    None
}

fn parse_pairs_correct(csv_file: &str) -> Vec<(String, String, bool)> {
    let mut pairs = Vec::new();

    println!("📖 CSV 파일 로드: {}", csv_file);

    if let Ok(file) = std::fs::File::open(csv_file) {
        let reader = std::io::BufReader::new(file);

        let mut success_count = 0;
        let mut failed_count = 0;
        let mut positive_pair = 0;
        let mut negative_pair = 0;

        for line in reader.lines() {
            if let Ok(line) = line {
                let parts: Vec<&str> = line.trim().split(',').collect();
                if parts.len() == 3 || (parts.len() == 4 && parts[3] == "") {
                    // 동일인 쌍: person, id1, id2
                    let person = parts[0].trim().to_string();
                    let id1 = parts[1].trim().to_string();
                    let id2 = parts[2].trim().to_string();

                    let dir_path = format!("lfw/lfw-deepfunneled/{}", person);

                    if let (Some(path1), Some(path2)) = (
                        find_image_file_with_padding(&dir_path, &id1, &person),
                        find_image_file_with_padding(&dir_path, &id2, &person),
                    ) {
                        if Path::new(&path1).exists() && Path::new(&path2).exists() {
                            pairs.push((path1, path2, true));
                            success_count += 1;
                            positive_pair += 1;
                        } else {
                            failed_count += 1;
                        }
                    } else {
                        eprintln!("fail {}", dir_path);
                        failed_count += 1;
                    }
                } else if parts.len() == 4 {
                    // 다인 쌍: person1, id1, person2, id2
                    let person1 = parts[0].trim().to_string();
                    let id1 = parts[1].trim().to_string();
                    let person2 = parts[2].trim().to_string();
                    let id2 = parts[3].trim().to_string();

                    let dir_path1 = format!("lfw/lfw-deepfunneled/{}", person1);
                    let dir_path2 = format!("lfw/lfw-deepfunneled/{}", person2);

                    if let (Some(path1), Some(path2)) = (
                        find_image_file_with_padding(&dir_path1, &id1, &person1),
                        find_image_file_with_padding(&dir_path2, &id2, &person2),
                    ) {
                        // 실제 파일 존재 여부 확인
                        if Path::new(&path1).exists() && Path::new(&path2).exists() {
                            pairs.push((path1, path2, false));
                            success_count += 1;
                            negative_pair += 1;
                        } else {
                            failed_count += 1;
                        }
                    } else {
                        failed_count += 1;
                    }
                }
            }
        }

        println!("✅ 로드 결과: 성공={}, 실패={} Positive={} Negative={}", success_count, failed_count, positive_pair, negative_pair);
    }

    pairs
}

/// 성능 평가 함수
fn evaluate_pairs(
    pairs: &[(String, String, bool)],
    embeddings: &HashMap<String, Vec<f32>>,
    threshold: f32,
) -> (f32, f32, f32, f32) {
    let mut tp = 0;  // True Positive (동일인, 정확히 감지)
    let mut tn = 0;  // True Negative (다인, 정확히 감지)
    let mut fp = 0;  // False Positive (다인, 동일인으로 잘못 감지)
    let mut fn_count = 0;  // False Negative (동일인, 다인으로 잘못 감지)

    for (path1, path2, is_same) in pairs {
        if let (Some(emb1), Some(emb2)) = (embeddings.get(path1), embeddings.get(path2)) {
            let similarity = cosine_similarity(emb1, emb2);
            let predicted_same = similarity > threshold;

            if *is_same && predicted_same {
                tp += 1;
            } else if !*is_same && !predicted_same {
                tn += 1;
            } else if !*is_same && predicted_same {
                fp += 1;
            } else if *is_same && !predicted_same {
                fn_count += 1;
            }
        }
    }

    let accuracy = (tp + tn) as f32 / pairs.len() as f32;
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (accuracy, precision, recall, f1)
}

fn main() -> std::io::Result<()> {
    // let args: Vec<String> = std::env::args().collect();
    //
    // if args.len() < 2 {
    //     println!("Usage: {} <input_size>", args[0]);
    //     println!("Example: {} 320", args[0]);
    //     println!("         {} 640", args[0]);
    //     return Ok(());
    // }
    //
    // let input_size: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(320);
    let input_size = 320;   // lfw image가    250x250 이므로 640은 적용하기 어려움

    println!("=== 얼굴 인식 성능 평가 ===");
    println!("Input size: {}", input_size);

    // 모델 로드
    let mut blaze = BlazeFaceModel::new("models/blaze.onnx", 0.5, 0.3, 2).unwrap();
    let mut scrfd = SCRFDDetector::new("models/det_500m.onnx", 0.5, 0.25, input_size).unwrap();
    let mut arcface = ArcFaceModel::new("models/w600k_mbf.onnx").unwrap();

    let pairs = parse_pairs_correct("lfw/pairs.csv");

    println!("총 쌍의 수: {}", pairs.len());

    // 모든 이미지에서 임베딩 추출
    let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
    let mut processed = 0;
    let mut failed = 0;

    for (path1, path2, _) in &pairs {
        for path in [path1, path2] {
            if !embeddings.contains_key(path) {
                let full_path = format!("{}", path);

                match extract_embedding(&full_path, &mut blaze, &mut scrfd, &mut arcface) {
                    Some(embedding) => {
                        embeddings.insert(path.clone(), embedding);
                        processed += 1;
                    }
                    None => {
                        eprintln!("⚠️ 임베딩 추출 실패: {}", full_path);
                        failed += 1;
                    }
                }

                if (processed + failed) % 100 == 0 {
                    println!("처리됨: {} / 실패: {}", processed, failed);
                }
            }
        }
    }

    println!("\n=== 임베딩 추출 완료 ===");
    println!("성공: {}", processed);
    println!("실패: {}", failed);

    // 성능 평가
    let thresholds = vec![0.4, 0.45, 0.5, 0.55, 0.6, 0.65];
    let mut results_file = File::create(format!("results_{}.csv", input_size))?;
    writeln!(results_file, "Threshold,Accuracy,Precision,Recall,F1")?;

    println!("\n=== 성능 평가 (Threshold별) ===");
    for threshold in thresholds {
        let (accuracy, precision, recall, f1) = evaluate_pairs(&pairs, &embeddings, threshold);
        println!("Threshold: {:.2} | Accuracy: {:.4} | Precision: {:.4} | Recall: {:.4} | F1: {:.4}",
                 threshold, accuracy, precision, recall, f1);
        writeln!(results_file, "{:.2},{:.4},{:.4},{:.4},{:.4}",
                 threshold, accuracy, precision, recall, f1)?;
    }

    println!("\n✅ 결과가 results_{}.csv에 저장되었습니다.", input_size);
    Ok(())
}
