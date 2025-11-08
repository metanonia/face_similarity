// Copyright (c) 2025 metanonia
//
// This source code is licensed under the MIT License.
// See the LICENSE file in the project root for license terms.

//! # scrfd_model
//!
//! This module implements face landmark detection using the SCRFD-500M model via ONNX Runtime.
// Models: det_500m.onnx

use opencv::{
    core::{self, Mat, Point2f, Rect, Size, CV_32F},
    imgproc,
};

use ndarray::Array4;
use std::error::Error;
use opencv::core::{no_array, AlgorithmHint, MatExprTraitConst, MatTrait, MatTraitConstManual, MatTraitManual, Rect2f, Scalar, Vector, BORDER_CONSTANT};
use opencv::imgproc::INTER_LINEAR;
use opencv::prelude::MatTraitConst;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;

#[derive(Clone)]
pub struct FaceDetection {
    pub bbox: Rect2f,
    pub confidence: f32,
    pub landmarks: Vec<Point2f>,
}

pub struct SCRFDDetector {
    session: Session,
    conf_threshold: f32,
    nms_threshold: f32,
    input_size: i32,
}

impl SCRFDDetector {
    pub fn new(
        model_path: &str,
        conf_threshold: f32,
        nms_threshold: f32,
        input_size: i32,
    ) -> Result<Self, Box<dyn Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            conf_threshold,
            nms_threshold,
            input_size,
        })
    }

    fn preprocess(&self, img: &Mat) -> Result<Tensor<f32>, Box<dyn Error>> {
        // 직접 리사이즈 (패딩 없음)
        let mut resized = Mat::default();
        imgproc::resize(
            img,
            &mut resized,
            Size::new(self.input_size, self.input_size),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // BGR -> RGB
        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &resized,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Mat -> ndarray (HWC -> CHW)
        let h = rgb.rows() as usize;
        let w = rgb.cols() as usize;

        let mut array = Array4::<f32>::zeros((1, 3, h, w));

        for y in 0..h {
            for x in 0..w {
                let pixel = rgb.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                // 정규화: (pixel - 127.5) / 128
                array[[0, 0, y, x]] = (pixel[0] as f32 - 127.5) / 128.0;
                array[[0, 1, y, x]] = (pixel[1] as f32 - 127.5) / 128.0;
                array[[0, 2, y, x]] = (pixel[2] as f32 - 127.5) / 128.0;
            }
        }

        let tensor: Tensor<f32> = Tensor::from_array(array.into_dyn())
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        Ok(tensor)
    }

    pub fn detect(&mut self, img: &Mat) -> Result<Vec<FaceDetection>, Box<dyn Error>> {
        let orig_h = img.rows();
        let orig_w = img.cols();

        // 전처리
        let input_tensor = self.preprocess(img)?;

        // 추론
        let outputs = self.session.run(inputs![input_tensor.clone()])?;

        let mut all_detections = Vec::new();

        // 각 stride별 처리 (8, 16, 32)
        let strides = vec![
            (8, 0, 3, 6),    // stride, conf_idx, bbox_idx, landmark_idx
            (16, 1, 4, 7),
            (32, 2, 5, 8),
        ];

        for (stride, conf_idx, bbox_idx, lm_idx) in strides {
            let detections = Self::process_stride(
                &outputs,
                stride,
                conf_idx,
                bbox_idx,
                lm_idx,
                orig_w,
                orig_h,
                self.input_size,
                self.conf_threshold,
            )?;

            all_detections.extend(detections);
        }

        // NMS 적용
        let filtered = Self::apply_nms(&all_detections, self.nms_threshold)?;

        // println!("검출: {} -> NMS 후: {}", all_detections.len(), filtered.len());

        Ok(filtered)
    }

    fn process_stride(
        outputs: &SessionOutputs,
        stride: i32,
        conf_idx: usize,
        bbox_idx: usize,
        lm_idx: usize,
        orig_w: i32,
        orig_h: i32,
        input_size: i32,
        conf_threshold: f32,
    ) -> Result<Vec<FaceDetection>, Box<dyn Error>> {
        let mut detections = Vec::new();

        let scores = outputs[conf_idx].try_extract_tensor::<f32>()?;
        let bboxes = outputs[bbox_idx].try_extract_tensor::<f32>()?;
        let landmarks = outputs[lm_idx].try_extract_tensor::<f32>()?;

        let score_data = scores.1;
        let bbox_data = bboxes.1;
        let landmark_data = landmarks.1;

        let feat_w = input_size / stride;
        let feat_h = input_size / stride;
        let num_grid_cells = (feat_w * feat_h) as usize;

        let num_anchors_per_cell = if score_data.len() > num_grid_cells {
            score_data.len() / num_grid_cells
        } else {
            1
        };

        let scale_factor_x = orig_w as f32 / input_size as f32;
        let scale_factor_y = orig_h as f32 / input_size as f32;

        for i in 0..score_data.len() {
            let score = score_data[i];
            if score < conf_threshold {
                continue;
            }

            let grid_cell_idx = i / num_anchors_per_cell;
            let anchor_in_cell = i % num_anchors_per_cell;

            let grid_y = (grid_cell_idx / feat_w as usize) as i32;
            let grid_x = (grid_cell_idx % feat_w as usize) as i32;

            let bbox_start = i * 4;
            if bbox_start + 4 > bbox_data.len() {
                continue;
            }

            let d_left = bbox_data[bbox_start];
            let d_top = bbox_data[bbox_start + 1];
            let d_right = bbox_data[bbox_start + 2];
            let d_bottom = bbox_data[bbox_start + 3];

            let cx = grid_x as f32 * stride as f32;
            let cy = grid_y as f32 * stride as f32;

            let x1 = (cx - d_left * stride as f32) * scale_factor_x;
            let y1 = (cy - d_top * stride as f32) * scale_factor_y;
            let x2 = (cx + d_right * stride as f32) * scale_factor_x;
            let y2 = (cy + d_bottom * stride as f32) * scale_factor_y;

            let w = (x2 - x1).max(1 as f32);
            let h = (y2 - y1).max(1 as f32);

            if w < 1 as f32 || h < 1 as f32 {
                continue;
            }

            // ✓ 랜드마크 범위 검사 추가
            let mut lms = Vec::new();
            let landmark_start = i * 10;

            if landmark_data.len() == score_data.len() * 10 {
                // 형태 1: [N, 10] (각 detection마다 10개 값)
                let landmark_start = i * 10;
                if landmark_start + 10 <= landmark_data.len() {
                    for j in 0..5 {
                        let lm_x = (cx + landmark_data[landmark_start + j * 2] * stride as f32) * scale_factor_x;
                        let lm_y = (cy + landmark_data[landmark_start + j * 2 + 1] * stride as f32) * scale_factor_y;
                        lms.push(Point2f::new(lm_x, lm_y));
                    }
                }
            } else {
                eprintln!("Unexpected landmark shape: scores={}, landmarks={}",
                          score_data.len(), landmark_data.len());
            }

            detections.push(FaceDetection {
                bbox: Rect2f::new(x1, y1, w, h),
                confidence: score,
                landmarks: lms,
            });
        }

        Ok(detections)
    }

    fn apply_nms(detections: &[FaceDetection], nms_threshold: f32) -> Result<Vec<FaceDetection>, Box<dyn Error>> {
        if detections.is_empty() {
            return Ok(Vec::new());
        }

        let mut sorted = detections.to_vec();
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();

        while !sorted.is_empty() {
            let current = sorted.remove(0);
            keep.push(current.clone());

            sorted.retain(|det| {
                let iou = Self::calculate_iou(&current.bbox, &det.bbox);
                iou < nms_threshold
            });
        }

        Ok(keep)
    }

    pub fn calculate_iou(box1: &Rect2f, box2: &Rect2f) -> f32 {
        let x1 = box1.x.max(box2.x);
        let y1 = box1.y.max(box2.y);
        let x2 = (box1.x + box1.width).min(box2.x + box2.width);
        let y2 = (box1.y + box1.height).min(box2.y + box2.height);

        let inter_w = (x2 - x1).max(0 as f32);
        let inter_h = (y2 - y1).max(0 as f32);
        let inter_area = (inter_w * inter_h) as f32;

        let box1_area = (box1.width * box1.height) as f32;
        let box2_area = (box2.width * box2.height) as f32;

        if box1_area + box2_area - inter_area <= 0.0 {
            return 0.0;
        }

        inter_area / (box1_area + box2_area - inter_area)
    }
}