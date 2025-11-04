use std::path::Path;
use opencv::{highgui, imgcodecs, imgproc, Result};
use opencv::core::{MatTraitConst, Rect, Size, Vector};
use opencv::prelude::Mat;
use crate::arcface_model::ArcFaceModel;
use crate::blaze_model::BlazeFaceModel;
use crate::face_align::FaceAlign;
use crate::scrfd_model::{SCRFDDetector};

mod blaze_model;
mod scrfd_model;
mod face_embedding_model;
mod arcface_model;
mod face_align;

fn get_filename_without_extension(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("image")
        .to_string()
}

fn main() -> Result<()>{
    let args: Vec<String> = std::env::args().collect();

    // 파라메터 확인
    if args.len() < 2 {
        println!("Usage: {} <image_path>", args[0]);
        println!("Example: {} ./images/face01.jpg", args[0]);
        return Ok(());
    }

    let image_path = &args[1];
    let image_filename = get_filename_without_extension(image_path);

    let blaze_input_size = 128;

    // 이미지 읽기
    let mut src = imgcodecs::imread(
        image_path,
        imgcodecs::IMREAD_COLOR
    )?;

    // 이미지가 제대로 읽혔는지 확인
    if src.empty() {
        println!("이미지를 읽을 수 없습니다!");
        return Ok(());
    }

    // 이미지 크기 (높이, 너비)
    let orig_height = src.rows() as f32;  // 이미지 높이(height)
    let orig_width = src.cols() as f32;  // 이미지 너비(width)
    println!("이미지 크기: {}x{}", orig_height, orig_width);

    // 채널 수
    let channels = src.channels();
    println!("채널 수: {}", channels);

    let mut face_detector = BlazeFaceModel::new("models/blaze.onnx", 0.5, 0.3, 2).unwrap();
    let mut landmark_detector = SCRFDDetector::new("models/det_500m.onnx", 0.5, 0.25).unwrap();
    let mut embedding_model =ArcFaceModel::new("models/w600k_mbf.onnx").unwrap();

    let mut resized = Mat::default();
    imgproc::resize(&src, &mut resized, Size::new(blaze_input_size, blaze_input_size), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

    let face_detections = face_detector.detect(&resized).unwrap();

    let mut face_index = 0;

    if face_detections.len() > 0 {
        for detection in face_detections {
            let color = opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0) ;

            let scale_x = orig_width / blaze_input_size as f32;
            let scale_y = orig_height / blaze_input_size as f32;

            let scaled_bbox = opencv::core::Rect::new(
                (detection.bbox.x as f32 * scale_x) as i32,
                (detection.bbox.y as f32 * scale_y) as i32,
                (detection.bbox.width as f32 * scale_x) as i32,
                (detection.bbox.height as f32 * scale_y) as i32,
            );

            // 바운딩 박스 확장 (예: 20% 여유)
            let margin_ratio = 0.2;
            let margin_x = (scaled_bbox.width as f32 * margin_ratio) as i32;
            let margin_y = (scaled_bbox.height as f32 * margin_ratio) as i32;

            let expanded_bbox = opencv::core::Rect::new(
                (scaled_bbox.x - margin_x).max(0),
                (scaled_bbox.y - margin_y).max(0),
                (scaled_bbox.width + 2 * margin_x).min(orig_width as i32 - scaled_bbox.x),
                (scaled_bbox.height + 2 * margin_y).min(orig_height as i32 - scaled_bbox.y),
            );

            // 확장된 영역으로 크롭
            let mut cropped = Mat::default();
            src.roi(expanded_bbox)?.copy_to(&mut cropped)?;
            let save_cropped_path = format!("images/cropped_{}_{}.png", image_filename, face_index);
            if let Err(e) = imgcodecs::imwrite(&save_cropped_path, &cropped, &Vector::new()) {
                eprintln!("⚠️ 크롭 이미지 저장 실패: {}", e);
            } else {
                println!("✅ 크롭 이미지 저장 완료: {}", save_cropped_path);
            }

            // 크롭된 이미지를 landmark 모델로 보냄
            let landmark_detects = landmark_detector.detect(&cropped).unwrap();
            println!("detected landmark {}", landmark_detects.len());

            let mut landmark_index = 0;
            for landmark in landmark_detects {
                let aligned = FaceAlign::norm_crop(&cropped, &landmark.landmarks, 112).unwrap();
                let save_path = format!("images/aligned_{}_{}_{}.png", image_filename, face_index, landmark_index);
                if let Err(e) = imgcodecs::imwrite(&save_path, &aligned, &Vector::new()) {
                    eprintln!("⚠️ 이미지 저장 실패: {}", e);
                } else {
                    println!("✅ Norm 얼굴 저장 완료: {}", save_path);
                }
                println!("confidence {:?}", landmark.confidence);
                println!("bbox {:?}", landmark.bbox);
                println!("landmarks {:?}", landmark.landmarks);

                // TOdo landmark src에 추가
                // 랜드마크 찍기
                for (i, lm) in landmark.landmarks.iter().enumerate() {
                    // 크롭 이미지 내 좌표에 크롭 위치 오프셋 추가
                    let lm_x = (lm.x as f32 + expanded_bbox.x as f32) as i32;
                    let lm_y = (lm.y as f32 + expanded_bbox.y as f32) as i32;

                    // 포인트 색 지정
                    let color = match i {
                        0 | 1 => opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0),       // 빨강 (눈)
                        2     => opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0),       // 파랑 (코)
                        3     => opencv::core::Scalar::new(255.0, 0.0, 255.0, 0.0),     // 마젠타 (입)
                        4 | 5 => opencv::core::Scalar::new(0.0, 165.0, 255.0, 0.0),     // 오렌지 (귀)
                        _     => opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),       // 그 외
                    };

                    // 원본 이미지에 동그라미 그리기
                    opencv::imgproc::circle(
                        &mut src,
                        opencv::core::Point::new(lm_x, lm_y),
                        4,            // 반지름
                        color,
                        -1,           // 채움
                        opencv::imgproc::LINE_8,
                        0,
                    ).unwrap();
                }


                // landmark를 이용하여 이미지 정렬
                let safe_bbox = Rect::new(0, 0, 112, 112);
                let embedded = embedding_model.embbeding(&aligned,safe_bbox).unwrap();
                // println!("{:?}", embedded);

                landmark_index += 1;
            }

            // 박스 그리기
            opencv::imgproc::rectangle(
                &mut src,
                scaled_bbox,
                color,
                2,
                opencv::imgproc::LINE_8,
                0
            ).unwrap();


            println!("Scaled box{:?}", scaled_bbox);

            face_index += 1;
        }
    } else {
        println!("얼굴이 감지되지 않았습니다.");
    }

    // 이미지 표시 (선택사항)
    // highgui::imshow("Landmark Visualization", &src)?;
    // highgui::wait_key(0)?;
    // highgui::destroy_all_windows()?;

    Ok(())
}
