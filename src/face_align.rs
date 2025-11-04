//! ArcFace Face Alignment Module
//!
//! Original sources: https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
//! Ported by: metanonia
//! Porting date: 2025-11-04
//!
//! This module provides face alignment functionality based on the ArcFace standard.
//! It includes:
//! - Face alignment using 5-point landmarks (norm_crop)
//! - Affine transformation matrix calculation (estimate_norm)
//! - Point transformation (trans_points_2d, trans_points_3d)
use opencv::{
    core::{self, Mat, Point2f, MatTrait},
    imgproc,
    calib3d::*,
    prelude::*,
};
use anyhow::Result;
use ndarray::{array, Array2};
use opencv::core::no_array;

const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

pub struct FaceAlign;

impl FaceAlign {
    /// ArcFace 기준 affine 변환 행렬 계산
    //
    /// # Arguments
    /// * `lmk` - 5개 얼굴 랜드마크 (Point2f 배열)
    /// * `image_size` - 출력 이미지 크기 (112 또는 128)
    ///
    /// # Returns
    /// * `Mat` - 2x3 Affine 변환 행렬
    pub fn estimate_norm(lmk: &[Point2f], image_size: i32) -> Result<Mat> {
        assert_eq!(lmk.len(), 5, "landmarks must have 5 points");
        assert!(image_size % 112 == 0 || image_size % 128 == 0);

        let (ratio, diff_x) = if image_size % 112 == 0 {
            (image_size as f32 / 112.0, 0.0)
        } else {
            (image_size as f32 / 128.0, 8.0 * (image_size as f32 / 128.0))
        };

        let mut dst: Vec<Point2f> = ARCFACE_DST
            .iter()
            .map(|&[x, y]| Point2f::new(x * ratio + diff_x, y * ratio))
            .collect();

        let src_mat = Self::points_to_mat(lmk)?;
        let dst_mat = Self::points_to_mat(&dst)?;

        let mut affine = estimate_affine_partial_2d(
            &src_mat,
            &dst_mat,
            &mut no_array(),
            RANSAC,
            5.0,
            2000,
            0.99,
            10,
        )?;

        // ★ 중요: CV_32F로 변환 확인
        if affine.typ() != core::CV_32F {
            let mut affine_f = Mat::default();
            affine.convert_to(&mut affine_f, core::CV_32F, 1.0, 0.0)?;
            affine = affine_f;
        }

        Ok(affine)
    }

    /// 정렬된 이미지만 반환
    pub fn norm_crop(img: &Mat, landmark: &[Point2f], image_size: i32) -> Result<Mat> {
        let M = Self::estimate_norm(landmark, image_size)?;
        let mut warped = Mat::default();
        imgproc::warp_affine(
            img,
            &mut warped,
            &M,
            core::Size::new(image_size, image_size),
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            core::Scalar::all(0.0),
        )?;
        Ok(warped)
    }

    /// 정렬된 이미지 + 변환 행렬 반환
    pub fn norm_crop2(img: &Mat, landmark: &[Point2f], image_size: i32) -> Result<(Mat, Mat)> {
        let M = Self::estimate_norm(landmark, image_size)?;
        let mut warped = Mat::default();
        imgproc::warp_affine(
            img,
            &mut warped,
            &M,
            core::Size::new(image_size, image_size),
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            core::Scalar::all(0.0),
        )?;
        Ok((warped, M))
    }

    /// 2D 포인트 변환
    pub fn trans_points_2d(pts: &[Point2f], M: &Mat) -> Result<Vec<Point2f>> {
        let mut new_pts = Vec::new();

        // M이 f32 타입인지 확인
        let M = if M.typ() != core::CV_32F {
            let mut M_f = Mat::default();
            M.convert_to(&mut M_f, core::CV_32F, 1.0, 0.0)?;
            M_f
        } else {
            M.clone()
        };

        for pt in pts {
            let x = pt.x;
            let y = pt.y;

            unsafe {
                // 첫 번째 행 (2x3 행렬이므로 3개 원소)
                let row0 = M.ptr(0)? as *const f32;
                let m00 = *row0 ;
                let m01 = *(row0.add(1));
                let m02 = *(row0.add(2));

                // 두 번째 행
                let row1 = M.ptr(1)? as *const f32;
                let m10 = *row1;
                let m11 = *(row1.add(1));
                let m12 = *(row1.add(2));

                let new_x = m00 * x + m01 * y + m02;
                let new_y = m10 * x + m11 * y + m12;

                new_pts.push(Point2f::new(new_x, new_y));
            }
        }

        Ok(new_pts)
    }

    /// 3D 포인트 변환 (x, y 변환, z는 스케일만 조정)
    pub fn trans_points_3d(pts: &[(f32, f32, f32)], M: &Mat) -> Result<Vec<(f32, f32, f32)>> {
        let mut new_pts = Vec::new();

        let M = if M.typ() != core::CV_32F {
            let mut M_f = Mat::default();
            M.convert_to(&mut M_f, core::CV_32F, 1.0, 0.0)?;
            M_f
        } else {
            M.clone()
        };

        unsafe {
            let row0 = M.ptr(0)? as *const f32;
            let row1 = M.ptr(1)? as *const f32;

            let m00 = *row0;
            let m01 = *(row0.add(1));

            let m10 = *row1;
            let m11 = *(row1.add(1));

            // 스케일 계산
            let scale = (m00 * m00 + m01 * m01).sqrt();

            for (x, y, z) in pts {
                let m02 = *(row0.add(2));
                let m12 = *(row1.add(2));

                let new_x = m00 * x + m01 * y + m02;
                let new_y = m10 * x + m11 * y + m12;
                let new_z = z * scale;

                new_pts.push((new_x, new_y, new_z));
            }
        }

        Ok(new_pts)
    }

    /// 포인트 변환 (2D 또는 3D 자동 판별)
    pub fn trans_points(pts: &[Point2f], M: &Mat) -> Result<Vec<Point2f>> {
        Self::trans_points_2d(pts, M)
    }

    /// Point2f 배열을 Mat으로 변환
    fn points_to_mat(points: &[Point2f]) -> Result<Mat> {
        let mut mat = Mat::zeros(points.len() as i32, 1, core::CV_32FC2)?.to_mat()?;

        for (i, pt) in points.iter().enumerate() {
            let mut row = mat.row_mut(i as i32)?;
            unsafe {
                let slice = row.data_typed_mut::<Point2f>()?;
                slice[0] = *pt;
            }
        }

        Ok(mat)
    }
}