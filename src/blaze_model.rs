// Copyright (c) 2025 metanonia
//
// This source code is licensed under the MIT License.
// See the LICENSE file in the project root for license terms.

//! # blaze_model
//!
//! This module implements face  detection using the blazeface model via ONNX Runtime.
//! Models: blaze.onnx

use opencv::core::{no_array, AlgorithmHint, Mat, Point2f, Scalar, Size, BORDER_CONSTANT};
use opencv::imgproc;
use opencv::prelude::*;
use std::error::Error;
use opencv::calib3d::{estimate_affine_partial_2d, RANSAC};
use ort::session::Session;
use ort::value::Tensor;

pub struct BlazeFaceModel {
    pub session: Session,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub max_detections: i64,
}

#[derive(Debug, Clone)]
pub struct BlazeFaceDetection {
    pub score: f32,
    pub bbox: opencv::core::Rect,
    pub landmarks: Vec<[f32; 2]>, // 6개의 landmarks (left_eye, right_eye, nose, mouth, left_ear, right_ear)
}

impl BlazeFaceModel {
    pub fn new(
        model_path: &str,
        conf_threshold: f32,
        iou_threshold: f32,
        max_detections: i64,
    ) -> Result<Self, Box<dyn Error>> {
        if !std::path::Path::new(model_path).exists() {
            return Err(format!("모델 파일을 찾을 수 없습니다: {}", model_path).into());
        }

        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            conf_threshold,
            iou_threshold,
            max_detections,
        })
    }

    fn preprocess(&self, img: &Mat) -> Result<Tensor<f32>, Box<dyn Error>> {
        // BGR 이미지를 RGB로 변환
        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &img,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // 128x128로 리사이즈
        let mut resized = Mat::default();
        imgproc::resize(
            &rgb,
            &mut resized,
            opencv::core::Size::new(128, 128),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let height = 128;
        let width = 128;
        let channels = 3;

        // CHW 형식의 f32 데이터를 Vec로 변환 (정규화 포함)
        let mut data = Vec::with_capacity(channels * height * width);

        // (1, 3, 128, 128) 형식으로 변환
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = *resized.at_2d::<opencv::core::Vec3b>(y as i32, x as i32)?;
                    // 0-255 범위를 0.0-1.0으로 정규화
                    data.push(pixel[c as usize] as f32 / 255.0);
                }
            }
        }

        // [1, 3, 128, 128] shape으로 텐서 생성
        let shape = vec![1, 3, 128, 128];
        let tensor = Tensor::<f32>::from_array((shape, data))?;
        Ok(tensor)
    }

    pub fn detect(&mut self, img: &Mat) -> Result<Vec<BlazeFaceDetection>, Box<dyn Error>> {
        let orig_height = img.rows() as f32;
        let orig_width = img.cols() as f32;

        // 이미지 전처리
        let image_tensor = self.preprocess(img)?;

        // threshold 텐서 생성
        let conf_threshold_tensor =
            Tensor::<f32>::from_array((vec![1], vec![self.conf_threshold]))?;
        let iou_threshold_tensor =
            Tensor::<f32>::from_array((vec![1], vec![self.iou_threshold]))?;
        let max_detections_tensor =
            Tensor::<i64>::from_array((vec![1], vec![self.max_detections]))?;

        // 모델 추론 실행
        let outputs = self.session.run(ort::inputs![
        "image" => image_tensor,
        "conf_threshold" => conf_threshold_tensor,
        "iou_threshold" => iou_threshold_tensor,
        "max_detections" => max_detections_tensor
    ])?;

        // 출력 개수 확인
        if outputs.len() < 1 {
            return Ok(Vec::new());
        }

        let boxes_value = &outputs[0];
        let boxes_shape: Vec<i64> = boxes_value.shape().iter().map(|&x| x as i64).collect();

        let num_faces = if boxes_shape.len() == 3 {
            if boxes_shape[1] == 0 {
                return Ok(Vec::new());
            }
            boxes_shape[1] as usize
        } else if boxes_shape.len() == 2 {
            if boxes_shape[0] == 0 {
                return Ok(Vec::new());
            }
            boxes_shape[0] as usize
        } else {
            return Err(format!("Unexpected boxes shape: {:?}", boxes_shape).into());
        };

        // 데이터 추출
        let boxes_result = boxes_value.try_extract_tensor::<f32>()?;
        let boxes_data = boxes_result.1;

        // scores 처리
        let scores_data = if outputs.len() > 1 {
            let scores_value = &outputs[1];
            let scores_shape: Vec<i64> = scores_value.shape().iter().map(|&x| x as i64).collect();

            let scores_count = if scores_shape.len() == 2 {
                scores_shape[1] as usize
            } else if scores_shape.len() == 1 {
                scores_shape[0] as usize
            } else {
                0
            };

            if scores_count == 0 {
                vec![1.0; num_faces]
            } else if let Ok(result) = scores_value.try_extract_tensor::<f32>() {
                result.1.to_vec()
            } else {
                vec![1.0; num_faces]
            }
        } else {
            vec![1.0; num_faces]
        };

        let mut detections = Vec::new();

        for i in 0..num_faces {
            let box_offset = i * 16;
            if box_offset + 15 >= boxes_data.len() {
                break;
            }

            let score = if i < scores_data.len() {
                scores_data[i]
            } else {
                1.0
            };

            if score < self.conf_threshold {
                continue;
            }

            // BBox 추출 - 형식: [y1, x1, y2, x2] (normalized 0-1)
            let y1_norm = boxes_data[box_offset];
            let x1_norm = boxes_data[box_offset + 1];
            let y2_norm = boxes_data[box_offset + 2];
            let x2_norm = boxes_data[box_offset + 3];

            // 원본 이미지 크기로 변환
            let x1 = (x1_norm * orig_width).max(0.0) as i32;
            let y1 = (y1_norm * orig_height).max(0.0) as i32;
            let x2 = (x2_norm * orig_width).min(orig_width) as i32;
            let y2 = (y2_norm * orig_height).min(orig_height) as i32;

            let width = x2 - x1;
            let height = y2 - y1;

            if width < 5 || height < 5 {
                continue;
            }

            let bbox = opencv::core::Rect::new(x1, y1, width, height);

            // 6개 랜드마크 추출 - 형식: [x, y] 쌍
            let mut landmarks = Vec::new();
            let landmark_indices = [
                (4, 5),   // left_eye
                (6, 7),   // right_eye
                (8, 9),   // nose
                (10, 11), // mouth
                (12, 13), // left_ear
                (14, 15), // right_ear
            ];

            for (x_idx, y_idx) in landmark_indices.iter() {
                let lm_x_norm = boxes_data[box_offset + x_idx];
                let lm_y_norm = boxes_data[box_offset + y_idx];

                landmarks.push([
                    lm_x_norm * orig_width,
                    lm_y_norm * orig_height,
                ]);
            }

            detections.push(BlazeFaceDetection {
                score,
                bbox,
                landmarks,
            });
        }

        Ok(detections)
    }

    /// BlazeFace 6점 (왼눈, 오른눈, 코, 입, 왼귀, 오른귀)을 사용한 얼굴 정렬
    pub fn align_face_6points_procrustes(
        &self,
        img: &Mat,
        detection: &BlazeFaceDetection,
        output_size: i32,
    ) -> Result<Mat, Box<dyn Error>> {
        if detection.landmarks.len() < 6 {
            return Err(
                "Not enough keypoints (need 6: right_eye, left_eye, nose, mouth, right_ear, left_ear)"
                    .into(),
            );
        }

        // 🔹 모든 랜드마크 유효성 검증
        for (i, lm) in detection.landmarks.iter().enumerate() {
            if !lm[0].is_finite() || !lm[1].is_finite() {
                return Err(format!("Landmark {} contains NaN or Inf", i).into());
            }
        }

        // 🔹 [f32; 2] → Point2f 변환 (모든 6개 점)
        let src_points_vec: Vec<Point2f> = detection
            .landmarks
            .iter()
            .map(|&p| Point2f::new(p[0], p[1]))
            .collect();
        let src_points_mat = Mat::from_slice(&src_points_vec)?;

        // 🔹 BlazeFace 6점 기준 표준 템플릿 (정규화 좌표)
        let std_points = vec![
            Point2f::new(0.62, 0.45), // 0: right_eye
            Point2f::new(0.38, 0.45), // 1: left_eye
            Point2f::new(0.50, 0.55), // 2: nose
            Point2f::new(0.50, 0.65), // 3: mouth
            Point2f::new(0.70, 0.50), // 4: right_ear
            Point2f::new(0.30, 0.50), // 5: left_ear
        ];

        // 🔹 출력 크기 기준으로 스케일링
        let dst_points: Vec<Point2f> = std_points
            .iter()
            .map(|p| Point2f::new(p.x * output_size as f32, p.y * output_size as f32))
            .collect();
        let dst_points_mat = Mat::from_slice(&dst_points)?;

        // 🔹 find_homography 사용 (6개 점 모두 활용)
        let mut mask = Mat::default();
        let homography = opencv::calib3d::find_homography(
            &src_points_mat,
            &dst_points_mat,
            &mut mask,
            opencv::calib3d::FM_RANSAC,
            4.0, // ransac_reproj_threshold
        )?;

        // 🔹 이미지 정렬 적용
        let mut aligned = Mat::default();
        imgproc::warp_perspective(
            img,
            &mut aligned,
            &homography,
            Size::new(output_size, output_size),
            imgproc::INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar::default(),
        )?;

        if aligned.empty() {
            return Err("Aligned image is empty".into());
        }

        Ok(aligned)
    }

    /// 얼굴 정렬을 위한 헬퍼 메서드
    pub fn align_face(
        &self,
        img: &Mat,
        detection: &BlazeFaceDetection,
        output_size: i32,
    ) -> Result<Mat, Box<dyn Error>> {
        if detection.landmarks.len() < 2 {
            return Err("랜드마크가 부족합니다".into());
        }

        // 🔹 랜드마크 검증
        for (i, lm) in detection.landmarks.iter().enumerate() {
            if !lm[0].is_finite() || !lm[1].is_finite() {
                return Err(format!("Landmark {} contains NaN or Inf", i).into());
            }
        }

        // 눈 위치
        let left_eye = detection.landmarks[0];
        let right_eye = detection.landmarks[1];

        // 두 눈 사이의 각도 계산
        let dy = right_eye[1] - left_eye[1];
        let dx = right_eye[0] - left_eye[0];
        let angle = dy.atan2(dx).to_degrees();

        // 두 눈의 중심점
        let eyes_center_x = (left_eye[0] + right_eye[0]) / 2.0;
        let eyes_center_y = (left_eye[1] + right_eye[1]) / 2.0;

        // 두 눈 사이의 거리 계산
        let dist = ((dx * dx) + (dy * dy)).sqrt();

        // 목표 눈 위치 설정 (출력 이미지 기준)
        let desired_left_eye_x = output_size as f32 * 0.35;
        let desired_right_eye_x = output_size as f32 * 0.65;
        let desired_eye_y = output_size as f32 * 0.35;

        // 목표 눈 사이 거리
        let desired_dist = desired_right_eye_x - desired_left_eye_x;

        // 스케일 계산
        let scale = desired_dist / dist;

        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("Invalid scale: {}", scale).into());
        }

        // 회전 행렬 생성 (눈 중심 기준, 각도, 스케일 포함)
        let eyes_center = opencv::core::Point2f::new(eyes_center_x, eyes_center_y);
        let mut rot_mat = imgproc::get_rotation_matrix_2d(eyes_center, angle as f64, scale as f64)?;

        // Translation 조정 (얼굴을 출력 이미지 중심으로 이동)
        let tx = output_size as f32 * 0.5;
        let ty = desired_eye_y;

        // rotation matrix의 translation 부분 수정
        *rot_mat.at_2d_mut::<f64>(0, 2)? += (tx as f64 - eyes_center_x as f64);
        *rot_mat.at_2d_mut::<f64>(1, 2)? += (ty as f64 - eyes_center_y as f64);

        // 🔹 정렬된 이미지가 제대로 생성되는지 확인
        let mut aligned = Mat::default();
        imgproc::warp_affine(
            img,
            &mut aligned,
            &rot_mat,
            opencv::core::Size::new(output_size, output_size),
            imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default(),
        )?;

        if aligned.empty() {
            return Err("Aligned image is empty".into());
        }

        Ok(aligned)
    }

    pub fn align_face_5points_procrustes(
        &self,
        img: &Mat,
        detection: &BlazeFaceDetection,
        output_size: i32,
    ) -> Result<Mat, Box<dyn Error>> {
        if detection.landmarks.len() < 6 {
            return Err("6개 랜드마크가 필요합니다".into());
        }

        // 🔹 모든 랜드마크 유효성 검증
        for (i, lm) in detection.landmarks.iter().enumerate() {
            if !lm[0].is_finite() || !lm[1].is_finite() {
                return Err(format!("Landmark {} contains NaN or Inf", i).into());
            }
        }

        // 5개 점 선택
        let src_pts = vec![
            opencv::core::Point2f::new(detection.landmarks[0][0], detection.landmarks[0][1]), // left_eye
            opencv::core::Point2f::new(detection.landmarks[1][0], detection.landmarks[1][1]), // right_eye
            opencv::core::Point2f::new(detection.landmarks[2][0], detection.landmarks[2][1]), // nose
            opencv::core::Point2f::new(detection.landmarks[4][0], detection.landmarks[4][1]), // left_ear
            opencv::core::Point2f::new(detection.landmarks[5][0], detection.landmarks[5][1]), // right_ear
        ];

        // 표준 좌표 (224x224 기준)
        let template_landmarks_224 = vec![
            (30.2946, 51.6963),   // left_eye
            (65.5318, 51.5014),   // right_eye
            (48.0252, 71.7366),   // nose
            (15.0, 65.0),         // left_ear
            (81.0, 65.0),         // right_ear
        ];

        // 동적 스케일 계산
        let left_eye_x = detection.landmarks[0][0];
        let right_eye_x = detection.landmarks[1][0];
        let actual_eye_dist = (right_eye_x - left_eye_x).abs();

        if actual_eye_dist < 5.0 {
            return Err(format!("Eye distance too small: {}", actual_eye_dist).into());
        }

        let template_eye_dist = 65.5318 - 30.2946;
        let scale = actual_eye_dist / template_eye_dist;

        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("Invalid scale: {}", scale).into());
        }

        // output_size에 맞게 스케일 조정
        let scale_to_output = output_size as f32 / 224.0;
        let mut dst_pts = Vec::new();

        for (x, y) in template_landmarks_224.iter() {
            dst_pts.push(opencv::core::Point2f::new(
                x * scale_to_output,
                y * scale_to_output,
            ));
        }

        // Vec<Point2f>를 Mat으로 변환
        let src_mat = opencv::core::Mat::from_slice(&src_pts)?;
        let dst_mat = opencv::core::Mat::from_slice(&dst_pts)?;

        // 🔹 올바른 시그니처로 호출
        let mut mask = Mat::default();
        let homography = opencv::calib3d::find_homography(
            &src_mat,
            &dst_mat,
            &mut mask,
            opencv::calib3d::FM_RANSAC,
            4.0,  // ransac_reproj_threshold
        )?;

        // 변환 적용
        let mut aligned = Mat::default();
        opencv::imgproc::warp_perspective(
            img,
            &mut aligned,
            &homography,
            opencv::core::Size::new(output_size, output_size),
            opencv::imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default(),
        )?;

        if aligned.empty() {
            return Err("Aligned image is empty".into());
        }

        Ok(aligned)
    }


    /// BlazeFace 6-point를 사용한 얼굴 정렬 (5점 기반)
    pub fn align_face_4points_procrustes(
        &self,
        img: &Mat,
        detection: &BlazeFaceDetection,
        output_size: i32,
    ) -> Result<Mat, Box<dyn Error>> {
        if detection.landmarks.len() < 6 {
            return Err("6개 랜드마크가 필요합니다".into());
        }

        // 🔹 모든 랜드마크 유효성 검증
        for (i, lm) in detection.landmarks.iter().enumerate() {
            if !lm[0].is_finite() || !lm[1].is_finite() {
                return Err(format!("Landmark {} contains NaN or Inf", i).into());
            }
        }

        // BlazeFace 6개 랜드마크
        // 0: left_eye, 1: right_eye, 2: nose, 3: mouth, 4: left_ear, 5: right_ear

        // 🔹 4개 점 선택 (get_perspective_transform은 정확히 4개만 받음)
        // left_eye, right_eye, left_ear, right_ear (4개 모서리)
        let src_pts = vec![
            opencv::core::Point2f::new(detection.landmarks[0][0], detection.landmarks[0][1]), // left_eye
            opencv::core::Point2f::new(detection.landmarks[1][0], detection.landmarks[1][1]), // right_eye
            opencv::core::Point2f::new(detection.landmarks[4][0], detection.landmarks[4][1]), // left_ear
            opencv::core::Point2f::new(detection.landmarks[5][0], detection.landmarks[5][1]), // right_ear
        ];

        // 🔹 동적 스케일 계산 (눈 사이 거리 기반)
        let left_eye_x = detection.landmarks[0][0];
        let right_eye_x = detection.landmarks[1][0];
        let actual_eye_dist = (right_eye_x - left_eye_x).abs();

        if actual_eye_dist < 5.0 {
            return Err(format!("Eye distance too small: {}", actual_eye_dist).into());
        }

        let template_eye_dist = 65.5318 - 30.2946; // 35.2372
        let scale = actual_eye_dist / template_eye_dist;

        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("Invalid scale: {}", scale).into());
        }

        // 🔹 표준 좌표 (224x224 기준, VGGFace2 표준)
        let template_landmarks_224 = vec![
            (30.2946, 51.6963),   // left_eye
            (65.5318, 51.5014),   // right_eye
            (15.0, 65.0),         // left_ear
            (81.0, 65.0),         // right_ear
        ];

        // output_size에 맞게 스케일 조정
        let scale_to_output = output_size as f32 / 224.0;
        let mut dst_pts = Vec::new();

        for (x, y) in template_landmarks_224.iter() {
            dst_pts.push(opencv::core::Point2f::new(
                x * scale_to_output,
                y * scale_to_output,
            ));
        }

        // 🔹 Vec<Point2f>를 Mat으로 변환 (정확히 4개 점)
        let src_mat = opencv::core::Mat::from_slice(&src_pts)?;
        let dst_mat = opencv::core::Mat::from_slice(&dst_pts)?;

        // Perspective 변환 행렬 계산 (4개 점 = 정확함)
        let perspective_matrix = opencv::imgproc::get_perspective_transform(&src_mat, &dst_mat, 0)?;

        // 변환 적용
        let mut aligned = Mat::default();
        opencv::imgproc::warp_perspective(
            img,
            &mut aligned,
            &perspective_matrix,
            opencv::core::Size::new(output_size, output_size),
            opencv::imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default(),
        )?;

        if aligned.empty() {
            return Err("Aligned image is empty".into());
        }

        Ok(aligned)
    }
}
