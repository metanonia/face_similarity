use std::io::Write;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use opencv::core::{Mat, MatTraitConst, Size};
use opencv::{highgui, imgcodecs, imgproc};
use rand::prelude::{IndexedRandom, StdRng};
use rand::SeedableRng;
use face_similarity::arcface_model::ArcFaceModel;
use face_similarity::blaze_model::BlazeFaceModel;
use face_similarity::face_align::FaceAlign;
use face_similarity::scrfd_model::SCRFDDetector;

/// 이미지 풀 경로 생성
fn make_full_path(filename: &str) -> String {
    format!("CelebA/Img/img_celeba/{}", filename)
}

/// 코사인 유사도 계산
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

/// 임베딩 불러오기 (경로 보정 포함)
fn load_embedding<'a>(path: &str, embeddings: &'a HashMap<String, Vec<f32>>) -> Option<&'a Vec<f32>> {
    let full_path = make_full_path(path);
    embeddings.get(&full_path)
}

/// 쌍 데이터 로드 (파일, 라벨)
fn load_pairs(file_path: &str, label: bool) -> Vec<(String, String, bool)> {
    let mut pairs = Vec::new();

    let file = File::open(file_path).expect(&format!("Failed to open {}", file_path));
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.trim().split(',').collect();
            if parts.len() == 2 {
                let path1 = parts[0].to_string();
                let path2 = parts[1].to_string();
                pairs.push((path1, path2, label));
            }
        }
    }
    pairs
}

/// 지정한 크기만큼 랜덤 샘플링 또는 전체 반환
fn sample_pairs(pairs: &[(String, String, bool)], sample_size: Option<usize>) -> Vec<(String, String, bool)> {
    if let Some(size) = sample_size {
        if size >= pairs.len() {
            return pairs.to_vec();
        }
        let mut rng = StdRng::seed_from_u64(42);
        pairs.choose_multiple(&mut rng, size).cloned().collect()
    } else {
        pairs.to_vec()
    }
}

/// TP, TN, FP, FN 개수 계산
fn evaluate_pairs(
    pairs: &[(String, String, bool)],
    embeddings: &HashMap<String, Vec<f32>>,
    threshold: f32,
) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (path1, path2, is_same) in pairs {
        if let (Some(emb1), Some(emb2)) = (load_embedding(path1, embeddings), load_embedding(path2, embeddings)) {
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
        } else {
            eprintln!("Warning: Missing embeddings for {} or {}", path1, path2);
        }
    }
    (tp, tn, fp, fn_count)
}


/// 이미지에서 임베딩 추출
fn extract_embedding(
    image_path: &str,
    blaze: &mut BlazeFaceModel,
    scrfd: &mut SCRFDDetector,
    arcface: &mut ArcFaceModel,
    use_alignment: bool,  // ✓ 정렬 여부 인자 추가
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

    // ✓ 정렬 여부에 따라 처리
    let aligned = if use_alignment {
        // SCRFD 랜드마크 감지
        let landmark_detects = scrfd.detect(&src).ok()?;
        if landmark_detects.is_empty() {
            eprintln!("No landmarks found");
            return None;
        }
        let landmark = &landmark_detects[0];

        // 정렬
        FaceAlign::norm_crop(&src, &landmark.landmarks, 112).ok()?
    } else {
        // 정렬 없이 직접 크롭한 이미지를 112x112로 리사이즈
        let mut resized_crop = Mat::default();
        imgproc::resize(&cropped, &mut resized_crop, Size::new(112, 112), 0.0, 0.0, imgproc::INTER_LINEAR).ok()?;
        resized_crop
    };

    let safe_bbox = opencv::core::Rect::new(0, 0, 112, 112);
    let embedding = arcface.embbeding(&aligned, safe_bbox).ok()?;
    Some(embedding)
}

// Accuracy, Precision, Recall, F1 계산 함수
fn evaluate_pairs_metrics(
    pairs: &[(String, String, bool)],
    embeddings: &HashMap<String, Vec<f32>>,
    threshold: f32,
) -> (f32, f32, f32, f32) {
    let (tp, tn, fp, fn_count) = evaluate_pairs(pairs, embeddings, threshold);

    let total = tp + tn + fp + fn_count;
    let accuracy = if total > 0 { (tp + tn) as f32 / total as f32 } else { 0.0 };
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
    // 명령행 인자: <프로그램> [샘플링 크기] [정렬 여부]
    let args: Vec<String> = env::args().collect();

    if args.len() < 1 {
        println!("Usage: {} [sample_size] [use_alignment: true/false]", args[0]);
        return Ok(());
    }

    let input_size: i32 = 320;

    let sample_size = if args.len() > 1 {
        args[1].parse::<usize>().ok()
    } else {
        None
    };

    // ✓ 얼굴 정렬 여부 (기본값: true)
    let use_alignment = if args.len() > 2 {
        args[2].to_lowercase() != "false"
    } else {
        true
    };

    println!("=== CelebA 얼굴 인식 성능 평가 ===");
    println!("Input size: {}", input_size);
    println!("Face alignment: {}", if use_alignment { "ON" } else { "OFF" });

    // 모델 로드
    let mut blaze = BlazeFaceModel::new("models/blaze.onnx", 0.5, 0.3, 2).unwrap();
    let mut scrfd = SCRFDDetector::new("models/det_500m.onnx", 0.5, 0.25, input_size).unwrap();
    let mut arcface = ArcFaceModel::new("models/w600k_mbf.onnx").unwrap();

    let mut positive_pairs = load_pairs("positive_pairs.txt", true);
    let mut negative_pairs = load_pairs("negative_pairs.txt", false);

    positive_pairs = sample_pairs(&positive_pairs, sample_size);
    negative_pairs = sample_pairs(&negative_pairs, sample_size);

    println!("Evaluating {} positive pairs and {} negative pairs", positive_pairs.len(), negative_pairs.len());

    // 모든 쌍에서 임베딩 추출
    let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
    let mut processed = 0;
    let mut failed = 0;

    let all_pairs: Vec<_> = positive_pairs.iter().chain(negative_pairs.iter()).collect();

    for (path1, path2, _) in all_pairs.iter() {
        for path in [path1, path2] {
            let full_path = make_full_path(path);

            if !embeddings.contains_key(&full_path) {
                // ✓ use_alignment 인자 전달
                match extract_embedding(&full_path, &mut blaze, &mut scrfd, &mut arcface, use_alignment) {
                    Some(embedding) => {
                        embeddings.insert(full_path.clone(), embedding);
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

    // ✓ 파일 이름에 정렬 여부 포함
    let alignment_str = if use_alignment { "aligned" } else { "no_align" };
    let mut results_file = File::create(format!("results_celeba_{}_{}.csv", input_size, alignment_str))?;
    writeln!(results_file, "Threshold,Accuracy,Precision,Recall,F1")?;

    println!("\n=== 성능 평가 (Threshold별) ===");
    for threshold in thresholds {
        let all_test_pairs: Vec<_> = positive_pairs.iter().chain(negative_pairs.iter()).cloned().collect();
        let (accuracy, precision, recall, f1) = evaluate_pairs_metrics(&all_test_pairs, &embeddings, threshold);

        println!("Threshold: {:.2} | Accuracy: {:.4} | Precision: {:.4} | Recall: {:.4} | F1: {:.4}",
                 threshold, accuracy, precision, recall, f1);
        writeln!(results_file, "{:.2},{:.4},{:.4},{:.4},{:.4}",
                 threshold, accuracy, precision, recall, f1)?;
    }

    println!("\n✅ 결과가 results_celeba_{}_{}.csv에 저장되었습니다.", input_size, alignment_str);
    Ok(())
}
