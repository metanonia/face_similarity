use std::env;
use opencv::{highgui, imgcodecs, imgproc, Result};
use opencv::core::{Mat, Point, Scalar, Vector};
use opencv::prelude::MatTraitConst;
use face_similarity::retina_model::{FaceDetection, RetinaFace};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 명령행 인자 가져오기
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("사용법: cargo run -- <이미지 경로>");
        std::process::exit(1);
    }

    let image_path = &args[1];
    println!("이미지 로드 중: {}", image_path);

    // 이미지 로드 (OpenCV)
    let img = opencv::imgcodecs::imread(image_path, opencv::imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        return Err(format!("이미지를 읽을 수 없습니다: {}", image_path).into());
    }

    println!("이미지 크기: {}x{}", img.cols(), img.rows());

    // RetinaFace 모델 로드
    let mut detector = RetinaFace::new(
        "models/retinaface-resnet50.onnx",  // 모델 경로 변경
        0.5,   // confidence threshold
        0.4,   // nms threshold
    )?;

    // 얼굴 검출
    let detections = detector.detect(&img)?;
    println!("검출된 얼굴: {} 개", detections.len());

    // 원본 이미지 복사
    let mut display_img = img.clone();

    // 랜드마크 그리기
    draw_landmarks(&mut display_img, &detections)?;

    // 이미지 표시
    highgui::named_window("RetinaFace Detection", highgui::WINDOW_NORMAL)?;
    highgui::imshow("RetinaFace Detection", &display_img)?;

    println!("🎬 이미지가 화면에 표시됩니다. 아무 키를 누르면 종료합니다.");
    highgui::wait_key(0)?;

    // 결과 이미지 저장
    // let output_path = "result.jpg";
    // imgcodecs::imwrite(output_path, &display_img, &Vector::new())?;
    // println!("✅ 결과 이미지 저장: {}", output_path);

    // 창 닫기
    highgui::destroy_all_windows()?;

    Ok(())
}

/// 랜드마크를 이미지에 그리기
pub fn draw_landmarks(img: &mut Mat, detections: &[FaceDetection]) -> opencv::Result<()> {
    for det in detections {
        // BBox 그리기
        imgproc::rectangle(
            img,
            det.bbox,
            Scalar::new(0.0, 255.0, 0.0, 0.0), // 초록색 (BGR)
            2,
            imgproc::LINE_8,
            0,
        )?;

        // Confidence 텍스트 그리기
        let conf_text = format!("Conf: {:.3}", det.confidence);
        imgproc::put_text(
            img,
            &conf_text,
            Point::new(det.bbox.x, det.bbox.y - 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;

        // 랜드마크 포인트 그리기 (5개)
        for (idx, landmark) in det.landmarks.iter().enumerate() {
            let point = Point::new(landmark.x as i32, landmark.y as i32);

            // 빨간 원으로 표시
            imgproc::circle(
                img,
                point,
                2,                              // 반지름
                Scalar::new(0.0, 0.0, 255.0, 0.0), // 빨간색 (BGR)
                -1,                             // 채우기 (-1)
                imgproc::LINE_8,
                0,
            )?;

            // 랜드마크 번호 표시
            // let text = format!("{}", idx);
            // imgproc::put_text(
            //     img,
            //     &text,
            //     Point::new(point.x + 8, point.y - 8),
            //     imgproc::FONT_HERSHEY_SIMPLEX,
            //     0.4,
            //     Scalar::new(255.0, 255.0, 255.0, 0.0), // 하얀색
            //     1,
            //     imgproc::LINE_8,
            //     false,
            // )?;
        }
    }

    Ok(())
}