use opencv::{
    core::{self, Mat, Point2f, Scalar, Size, Vector},
    dnn::{self, Net},
    imgproc,
    prelude::*,
};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: core::Rect,
    pub confidence: f32,
    pub landmarks: Vec<Point2f>, // 5개 점
}

#[derive(Debug, Clone)]
struct PriorBox {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

pub struct RetinaFace {
    net: Net,
    input_size: (i32, i32),
    conf_threshold: f32,
    priors: Vec<PriorBox>,
    variance: Vec<f32>,
}

impl RetinaFace {
    pub fn new(model_path: &str, conf_threshold: f32, _nms_threshold: f32, input_val: i32) -> Result<Self, Box<dyn Error>> {
        if !std::path::Path::new(model_path).exists() {
            return Err(format!("모델 파일을 찾을 수 없습니다: {}", model_path).into());
        }

        println!("RetinaFace 모델 로드 중: {}", model_path);
        let net = dnn::read_net_from_onnx(model_path)?;
        println!("RetinaFace 모델 로드 완료!");

        let input_size = (input_val, input_val);
        let priors = Self::generate_priors(input_size.0);
        let variance = vec![0.1, 0.2];

        // println!("Prior-box 생성 완료: {} 개", priors.len());

        Ok(Self {
            net,
            input_size,
            conf_threshold,
            priors,
            variance,
        })
    }

    /// Prior-box 생성 (16,800개)
    fn generate_priors(input_size: i32) -> Vec<PriorBox> {
        let mut priors = Vec::new();

        // (stride, anchor_sizes)
        let configs = vec![
            (8, vec![16, 32]),      // 80x80 -> 12,800
            (16, vec![64, 128]),    // 40x40  -> 3,200
            (32, vec![256, 512]),   // 20x20  -> 800
        ];

        for (stride, anchor_sizes) in configs {
            let feature_map_size = input_size / stride;

            for i in 0..feature_map_size {
                for j in 0..feature_map_size {
                    let cx = (j as f32 + 0.5) * stride as f32 / input_size as f32;
                    let cy = (i as f32 + 0.5) * stride as f32 / input_size as f32;

                    for &anchor_size in &anchor_sizes {
                        let w = anchor_size as f32 / input_size as f32;
                        let h = anchor_size as f32 / input_size as f32;

                        priors.push(PriorBox {
                            cx,
                            cy,
                            w,
                            h,
                        });
                    }
                }
            }
        }

        priors
    }

    fn decode_box(&self, loc: &[f32; 4], prior: &PriorBox) -> (f32, f32, f32, f32) {
        let variance = &self.variance;

        // 공식 RetinaFace 디코딩 로직
        let cx = prior.cx + loc[0] * variance[0] * prior.w;
        let cy = prior.cy + loc[1] * variance[0] * prior.h;
        let w  = prior.w * (loc[2] * variance[1]).exp();
        let h  = prior.h * (loc[3] * variance[1]).exp();

        let x1 = (cx - w / 2.0).clamp(0.0, 1.0);
        let y1 = (cy - h / 2.0).clamp(0.0, 1.0);
        let x2 = (cx + w / 2.0).clamp(0.0, 1.0);
        let y2 = (cy + h / 2.0).clamp(0.0, 1.0);

        (x1, y1, x2, y2)
    }


    /// 랜드마크 디코딩
    fn decode_landmarks(&self, landmarks: &[f32; 10], prior: &PriorBox) -> Vec<Point2f> {
        let variance = &self.variance;
        let mut points = Vec::new();

        for i in 0..5 {
            let x = landmarks[i * 2] * variance[0] * prior.w + prior.cx;
            let y = landmarks[i * 2 + 1] * variance[0] * prior.h + prior.cy;

            points.push(Point2f::new(
                x.max(0.0).min(1.0),
                y.max(0.0).min(1.0),
            ));
        }

        points
    }

    fn preprocess(&self, img: &Mat) -> Result<Mat, Box<dyn Error>> {
        // RetinaFace 전처리
        // swapRB=true로 BGR -> RGB 변환
        let mean = Scalar::new(123.0, 117.0, 104.0, 0.0); // RGB 순서
        let blob = dnn::blob_from_image(
            img,
            1.0,
            Size::new(self.input_size.0, self.input_size.1),
            mean,
            true,  // swapRB=true (BGR -> RGB)
            false, // crop=false
            core::CV_32F,
        )?;
        Ok(blob)
    }

    pub fn detect(&mut self, img: &Mat) -> Result<Vec<FaceDetection>, Box<dyn Error>> {
        let orig_h = img.rows();
        let orig_w = img.cols();

        let blob = self.preprocess(img)?;
        self.net.set_input(&blob, "", 1.0, Scalar::default())?;

        //  출력 레이어 이름 가져오기
        let output_layer_names = self.net.get_unconnected_out_layers_names()?;

        // println!("출력 레이어 이름들:");
        // for name in output_layer_names.iter() {
        //     println!("  - {}", name);
        // }

        // forward 호출 (레이어 이름 사용)
        let mut outputs = Vector::<Mat>::new();
        self.net.forward(
            &mut outputs,
            &output_layer_names,
        )?;

        // println!("총 출력 개수: {}", outputs.len());

        if outputs.len() < 3 {
            return Err(format!("예상한 3개 출력이 아닌 {}개 받음", outputs.len()).into());
        }

        // 출력 확인 및 매핑
        let mut bbox_mat: Option<Mat> = None;
        let mut conf_mat: Option<Mat> = None;
        let mut landmark_mat: Option<Mat> = None;

        for (i, name) in output_layer_names.iter().enumerate() {
            let mat = outputs.get(i)?;
            let shape = mat.mat_size();
            // println!("Output {}: name='{}', shape={:?}", i, name, shape);

            // 이름으로 매핑
            if name.contains("bbox") {
                bbox_mat = Some(mat.clone());
            } else if name.contains("confidence") {
                conf_mat = Some(mat.clone());
            } else if name.contains("landmark") {
                landmark_mat = Some(mat.clone());
            }
        }

        let bbox_mat = bbox_mat.ok_or("bbox 출력을 찾을 수 없음")?;
        let conf_mat = conf_mat.ok_or("confidence 출력을 찾을 수 없음")?;
        let landmark_mat = landmark_mat.ok_or("landmark 출력을 찾을 수 없음")?;

        let bbox_shape = bbox_mat.mat_size();
        let conf_shape = conf_mat.mat_size();
        let landmark_shape = landmark_mat.mat_size();

        // println!("📊 최종 출력 shape:");
        // println!("  bbox: {:?}", bbox_shape);
        // println!("  confidence: {:?}", conf_shape);
        // println!("  landmark: {:?}", landmark_shape);

        let num_detections = bbox_shape[1] as usize;

        // println!("검출된 앵커 개수: {}", num_detections);

        if num_detections == 0 {
            return Ok(Vec::new());
        }

        if num_detections != self.priors.len() {
            println!("⚠️ 경고: 검출 개수 {} != prior-box 개수 {}",
                     num_detections, self.priors.len());
        }

        let num_detections = num_detections.min(self.priors.len());
        let mut detections = Vec::new();

        for i in 0..num_detections {
            let bg_score = *conf_mat.at_3d::<f32>(0, i as i32, 0)?;
            let face_score = *conf_mat.at_3d::<f32>(0, i as i32, 1)?;

            if face_score < self.conf_threshold {
                continue;
            }

            let mut loc = [0.0f32; 4];
            for j in 0..4 {
                loc[j] = *bbox_mat.at_3d::<f32>(0, i as i32, j as i32)?;
            }

            let (x1, y1, x2, y2) = self.decode_box(&loc, &self.priors[i]);

            let x1_scaled = (x1 * orig_w as f32).max(0.0) as i32;
            let y1_scaled = (y1 * orig_h as f32).max(0.0) as i32;
            let x2_scaled = (x2 * orig_w as f32).min(orig_w as f32) as i32;
            let y2_scaled = (y2 * orig_h as f32).min(orig_h as f32) as i32;

            let w = x2_scaled - x1_scaled;
            let h = y2_scaled - y1_scaled;

            if w <= 0 || h <= 0 {
                continue;
            }

            let mut lm = [0.0f32; 10];
            for j in 0..10 {
                lm[j] = *landmark_mat.at_3d::<f32>(0, i as i32, j as i32)?;
            }

            let mut landmarks = self.decode_landmarks(&lm, &self.priors[i]);

            for point in &mut landmarks {
                point.x = (point.x * orig_w as f32).max(0.0).min(orig_w as f32);
                point.y = (point.y * orig_h as f32).max(0.0).min(orig_h as f32);
            }

            detections.push(FaceDetection {
                bbox: core::Rect::new(x1_scaled, y1_scaled, w, h),
                confidence: face_score,
                landmarks,
            });

            // if detections.len() <= 5 {
            //     println!("  Face {}: conf={:.3}, bbox=({},{},{},{})",
            //              i, face_score, x1_scaled, y1_scaled, w, h);
            // }
        }

        // println!("임계값 이상 검출: {} 개", detections.len());

        let detections = self.apply_nms(detections, 0.4)?;
        // println!("✅ NMS 후 최종: {} 개", detections.len());

        Ok(detections)
    }



    /// NMS (Non-Maximum Suppression)
    fn apply_nms(&self, mut detections: Vec<FaceDetection>, nms_threshold: f32) -> Result<Vec<FaceDetection>, Box<dyn Error>> {
        if detections.is_empty() {
            return Ok(detections);
        }

        // confidence 기준 내림차순 정렬
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut indices: Vec<usize> = (0..detections.len()).collect();

        while !indices.is_empty() {
            let current = indices[0];
            keep.push(detections[current].clone());

            if indices.len() == 1 {
                break;
            }

            indices.remove(0);
            let mut to_remove = Vec::new();

            for &idx in &indices {
                let iou = self.calculate_iou(&detections[current].bbox, &detections[idx].bbox);
                if iou > nms_threshold {
                    to_remove.push(idx);
                }
            }

            indices.retain(|idx| !to_remove.contains(idx));
        }

        Ok(keep)
    }

    /// IoU (Intersection over Union) 계산
    fn calculate_iou(&self, box1: &core::Rect, box2: &core::Rect) -> f32 {
        let x1_inter = box1.x.max(box2.x);
        let y1_inter = box1.y.max(box2.y);
        let x2_inter = (box1.x + box1.width).min(box2.x + box2.width);
        let y2_inter = (box1.y + box1.height).min(box2.y + box2.height);

        if x2_inter <= x1_inter || y2_inter <= y1_inter {
            return 0.0;
        }

        let inter_area = ((x2_inter - x1_inter) * (y2_inter - y1_inter)) as f32;
        let area1 = (box1.width * box1.height) as f32;
        let area2 = (box2.width * box2.height) as f32;
        let union_area = area1 + area2 - inter_area;

        inter_area / union_area
    }

    /// 얼굴 정렬 함수
    pub fn align_face(&self, img: &Mat, landmarks: &[Point2f], output_size: Size) -> opencv::Result<Mat> {
        if landmarks.len() != 5 {
            return Err(opencv::Error::new(
                core::StsError,
                format!("랜드마크는 5개여야 합니다. 현재: {}", landmarks.len()),
            ));
        }

        // 기준 랜드마크 (112x112 기준)
        let dst_landmarks = vec![
            Point2f::new(38.2946, 51.6963), // 왼쪽 눈
            Point2f::new(73.5318, 51.5014), // 오른쪽 눈
            Point2f::new(56.0252, 71.7366), // 코
            Point2f::new(41.5493, 92.3655), // 왼쪽 입
            Point2f::new(70.7299, 92.2041), // 오른쪽 입
        ];

        // 출력 크기에 맞춰 스케일링
        let scale_x = output_size.width as f32 / 112.0;
        let scale_y = output_size.height as f32 / 112.0;

        let scaled_dst: Vec<Point2f> = dst_landmarks.iter()
            .map(|p| Point2f::new(p.x * scale_x, p.y * scale_y))
            .collect();

        // 5점 중 3점 사용 (눈 2개 + 코)
        let src_pts: [Point2f; 3] = [landmarks[0], landmarks[1], landmarks[2]];
        let dst_pts: [Point2f; 3] = [scaled_dst[0], scaled_dst[1], scaled_dst[2]];

        let src_vector = Vector::from_slice(&src_pts);
        let dst_vector = Vector::from_slice(&dst_pts);

        // Affine 변환 행렬 계산
        let transform = imgproc::get_affine_transform(&src_vector, &dst_vector)?;

        // 이미지 변환
        let mut aligned = Mat::default();
        imgproc::warp_affine(
            img,
            &mut aligned,
            &transform,
            output_size,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::default(),
        )?;

        Ok(aligned)
    }
}
