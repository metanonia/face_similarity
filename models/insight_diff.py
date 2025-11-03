from insightface.app import FaceAnalysis
import cv2
import numpy as np

# 얼굴 정렬 함수 (5점 랜드마크 기준)
def align_face(img, landmarks):
    src = np.array([
        [38.2946, 51.6963],  # 왼쪽 눈
        [73.5318, 51.5014],  # 오른쪽 눈
        [56.0252, 71.7366],  # 코끝
        [41.5493, 92.3655],  # 왼쪽 입꼬리
        [70.7299, 92.2041]   # 오른쪽 입꼬리
    ], dtype=np.float32)

    dst = landmarks.astype(np.float32)

    tform = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
    aligned_face = cv2.warpAffine(img, tform, (112, 112), borderValue=0.0)

    return aligned_face

# InsightFace 준비
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(320,320))

# 원본 이미지와 탐지
img_orig = cv2.imread("../images/cropped_face.jpg")
faces_orig = app.get(img_orig)

if len(faces_orig) == 0:
    print("원본 이미지에서 얼굴을 찾지 못했습니다.")
else:
    face_orig = faces_orig[0]

    # InsightFace 얼굴 랜드마크로 직접 정렬 이미지 생성
    aligned_from_orig = align_face(img_orig, face_orig.kps)
    cv2.imwrite("aligned_from_insightface.png", aligned_from_orig)  # 확인용 저장

    # 정렬 이미지 탐지
    img_aligned = cv2.imread("../images/aligned_face_112.png")
    faces_aligned = app.get(img_aligned)

    if len(faces_aligned) == 0:
        print("정렬된 이미지에서 얼굴을 찾지 못했습니다.")
    else:
        # 두 임베딩 추출
        embedding_aligned = faces_aligned[0].embedding
        embedding_new_aligned = app.get(aligned_from_orig)[0].embedding

        # 원본에서 정렬 결과 임베딩과 코사인 유사도 계산
        cos_sim_1 = np.dot(face_orig.embedding, embedding_new_aligned) / (np.linalg.norm(face_orig.embedding) * np.linalg.norm(embedding_new_aligned))

        # 사용자가 만든 정렬 이미지 임베딩과 코사인 유사도 계산
        cos_sim_2 = np.dot(embedding_new_aligned, embedding_aligned) / (np.linalg.norm(embedding_new_aligned) * np.linalg.norm(embedding_aligned))

        print("원본->내가 만든 정렬 이미지 임베딩 코사인 유사도:", cos_sim_1)
        print("내가 만든 정렬 이미지와 사용자 정렬 이미지 임베딩 코사인 유사도:", cos_sim_2)

