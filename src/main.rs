use opencv::{highgui, imgcodecs, imgproc, Result};
use opencv::core::{MatTraitConst, Rect, Size, Vector};
use opencv::prelude::Mat;
use crate::arcface_model::ArcFaceModel;
use crate::blaze_model::BlazeFaceModel;
use crate::scrfd_model::{FaceAligner, SCRFDDetector};

mod blaze_model;
mod scrfd_model;
mod face_embedding_model;
mod arcface_model;

fn main() -> Result<()>{
    let BlazeInputSize = 128;
    let ScrdfInputSize = 320;

    // 이미지 읽기
    let mut src = imgcodecs::imread(
        "./images/face01.jpg",
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
    imgproc::resize(&src, &mut resized, Size::new(BlazeInputSize, BlazeInputSize), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

    let face_detections = face_detector.detect(&resized).unwrap();
    if face_detections.len() > 0 {
        for detection in face_detections {
            let color = opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0) ;

            let scale_x = orig_width / BlazeInputSize as f32;
            let scale_y = orig_height / BlazeInputSize as f32;

            let scaled_bbox = opencv::core::Rect::new(
                (detection.bbox.x as f32 * scale_x) as i32,
                (detection.bbox.y as f32 * scale_y) as i32,
                (detection.bbox.width as f32 * scale_x) as i32,
                (detection.bbox.height as f32 * scale_y) as i32,
            );



            // 랜드마크 찍기
            // for (i, lm) in detection.landmarks.iter().enumerate() {
            //     let lm_x = (lm[0] * scale_x) as i32;
            //     let lm_y = (lm[1] * scale_y) as i32;
            //
            //     let color = match i {
            //         0 | 1 => opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0),       // 빨강 (눈)
            //         2     => opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0),       // 파랑 (코)
            //         3     => opencv::core::Scalar::new(255.0, 0.0, 255.0, 0.0),     // 마젠타 (입)
            //         4 | 5 => opencv::core::Scalar::new(0.0, 165.0, 255.0, 0.0),     // 오렌지 (귀)
            //         _     => opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),       // 나머지
            //     };
            //
            //     opencv::imgproc::circle(
            //         &mut src,
            //         opencv::core::Point::new(lm_x, lm_y),
            //         4,            // 크기
            //         color,
            //         -1,           // 채움
            //         opencv::imgproc::LINE_8,
            //         0,
            //     ).unwrap();
            // }
            //
            // let text = format!("Score: {:.2}", detection.score);
            // opencv::imgproc::put_text(
            //     &mut src,
            //     &text,
            //     opencv::core::Point::new(scaled_bbox.x, scaled_bbox.y - 10),
            //     opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            //     0.6,
            //     opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            //     2,
            //     opencv::imgproc::LINE_8,
            //     false
            // ).unwrap();

            // 원본 이미지에서 지정된 영역 추출
            // let mut cropped = Mat::default();
            // src.roi(scaled_bbox)?.copy_to(&mut cropped)?;
            // highgui::imshow("Image", &cropped)?;
            // highgui::wait_key(0)?;
            // highgui::destroy_all_windows()?;
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
            let save_cropped_path = "images/cropped_face1.png";
            if let Err(e) = imgcodecs::imwrite(save_cropped_path, &cropped, &opencv::core::Vector::new()) {
                eprintln!("⚠️ 크롭 이미지 저장 실패: {}", e);
            } else {
                println!("✅ 크롭 이미지 저장 완료: {}", save_cropped_path);
            }



            // 크롭된 이미지를 landmark 모델로 보냄
            let landmark_detects = landmark_detector.detect(&cropped).unwrap();
            println!("detected landmark {}", landmark_detects.len());
            for landmark in landmark_detects {
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
                let aligned = FaceAligner::align_face(&cropped, &landmark.landmarks, 112).unwrap();
                let save_path = "images/aligned_face1.png";
                if let Err(e) = imgcodecs::imwrite(save_path, &aligned, &opencv::core::Vector::new()) {
                    eprintln!("⚠️ 이미지 저장 실패: {}", e);
                } else {
                    println!("✅ Aligned 얼굴 저장 완료: {}", save_path);
                }
                let embedded = embedding_model.embbeding(&aligned,safe_bbox).unwrap();
                // println!("{:?}", embedded);
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
        }
    }

    // 이미지 표시 (선택사항)
    opencv::highgui::imshow("Landmark Visualization", &src)?;
    opencv::highgui::wait_key(0)?;
    opencv::highgui::destroy_all_windows()?;


    Ok(())
}
