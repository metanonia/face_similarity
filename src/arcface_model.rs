use opencv::{
    core::{Mat, Scalar, Size, Vector},
    imgcodecs, imgproc,
    prelude::*,
};
use std::error::Error;
use std::fmt;
use ndarray::Array4;
use opencv::core::{AlgorithmHint, Point2f, Rect};
use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;

#[derive(Debug)]
struct ShapeError {
    shape: Vec<usize>,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid tensor shape: {:?}", self.shape)
    }
}

impl Error for ShapeError {}


pub struct ArcFaceModel {
    pub session: Session,
    pub input_size: i32,
    pub conf_threshold: f32,
}

impl ArcFaceModel {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        if !std::path::Path::new(model_path).exists() {
            return Err(format!("모델 파일을 찾을 수 없습니다: {}", model_path).into());
        }

        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session: session,
            input_size: 320,
            conf_threshold: 0.25,
        })
    }

    pub fn embbeding(&mut self, img: &Mat, safe_bbox: Rect) -> Result<Vec<(f32)>, Box<dyn Error>> {
        let face_roi = match Mat::roi(img, safe_bbox) {
            Ok(roi) => roi,
            Err(e) => {
                println!("⚠️ 얼굴  ROI 추출 실패 - {}",  e);
                return Err(e.into()); // 또는 return Err(Box::new(e));
            }
        };

        let mut face_crop_resized = Mat::default();
        if let Err(e) = imgproc::resize(&face_roi, &mut face_crop_resized, Size::new(112, 112), 0.0, 0.0, imgproc::INTER_LINEAR) {
            println!("⚠️ 얼굴: 리사이즈 실패 - {}",  e);
            return Err(e.into()); // 또는 return Err(Box::new(e));
        }

        let mut face_crop_rgb = Mat::default();
        if let Err(e) = imgproc::cvt_color(&face_crop_resized, &mut face_crop_rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT) {
            println!("⚠️ 얼굴: 색상 변환 실패 - {}", e);
            return Err(e.into()); // 또는 return Err(Box::new(e));
        }

        // ✅ Mat -> Vec<f32>로 변환 + 정규화
        use opencv::core::Vec3b;
        let rows = face_crop_rgb.rows();
        let cols = face_crop_rgb.cols();
        let mut face_input_vec = Vec::with_capacity((rows * cols * 3) as usize);

        for y_coord in 0..rows {
            for x_coord in 0..cols {
                let pixel: Vec3b = *face_crop_rgb.at_2d::<Vec3b>(y_coord, x_coord)?;
                face_input_vec.push((pixel[0] as f32 - 127.5) / 128.0);
                face_input_vec.push((pixel[1] as f32 - 127.5) / 128.0);
                face_input_vec.push((pixel[2] as f32 - 127.5) / 128.0);
            }
        }

        // ✅ 텐서 생성
        let face_input_shape = vec![1, 112, 112, 3];
        let mut array = Array4::<f32>::zeros((1, 3, 112, 112)); // NCHW
        for y in 0..112 {
            for x in 0..112 {
                let idx = (y * 112 + x) * 3;
                array[[0, 0, y, x]] = face_input_vec[idx];     // R
                array[[0, 1, y, x]] = face_input_vec[idx + 1]; // G
                array[[0, 2, y, x]] = face_input_vec[idx + 2]; // B
            }
        }
        let face_input_tensor = match Tensor::<f32>::from_array((array.into_dyn())) {
            Ok(tensor) => tensor,
            Err(e) => {
                println!("⚠️ 얼굴: 텐서 생성 실패 - {}", e);
                return Err(e.into()); // 또는 return Err(Box::new(e));
            }
        };

        /// ✅ ArcFace 추론
        // println!("🧠 얼굴 #{}: ArcFace 추론 시작", idx);
        let emb_outputs = match self.session.run(inputs![face_input_tensor]) {
            Ok(outputs) => outputs,
            Err(e) => {
                println!("⚠️ 얼굴: 임베딩 생성 실패 - {}", e);
                return Err(e.into()); // 또는 return Err(Box::new(e));
            }
        };

        let emb_tensor_out = match emb_outputs[0].try_extract_tensor::<f32>() {
            Ok(tensor) => tensor,
            Err(e) => {
                println!("⚠️ 얼굴: 임베딩 추출 실패 - {}", e);
                return Err(e.into()); // 또는 return Err(Box::new(e));
            }
        };

        let mut embedding: Vec<f32> = emb_tensor_out.1.to_vec();
        Self::normalize_l2(&mut embedding);
        Ok(embedding)
    }

    fn normalize_l2(vec: &mut Vec<f32>) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
    }
}