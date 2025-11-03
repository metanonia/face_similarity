from insightface.app import FaceAnalysis
import cv2
import numpy as np
import onnxruntime as ort

# 얼굴 정렬 함수 (5점 랜드마크 기준)
# ArcFace 임베딩 모델 ONNX 경로
arcface_model_path = "w600k_mbf.onnx"
ort_session = ort.InferenceSession(arcface_model_path)

def preprocess_aligned_face_already_112(img):
    """
    이미 112x112로 정렬된 얼굴에 대해 전처리 수행 (RGB 변환, 정규화, 차원 변환)
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transp = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_transp, axis=0)
    return img_batch

def get_embedding_from_aligned_image(img_path, ort_session):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")

    input_tensor = preprocess_aligned_face_already_112(img)

    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_tensor})
    embedding = outputs[0][0]
    return embedding

# InsightFace 준비
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(320,320))

# 원본 이미지와 탐지
img_orig = cv2.imread("../images/cropped_face.jpg")
faces_orig = app.get(img_orig)
img_orig2 = cv2.imread("../images/cropped_face2.png")
faces_orig2 = app.get(img_orig2)
img_orig3 = cv2.imread("../images/cropped_face3.png")
faces_orig3 = app.get(img_orig3)

emb_1 = faces_orig[0].embedding
emb_2 = faces_orig2[0].embedding
emb_3 = faces_orig3[0].embedding


if len(faces_orig) == 0:
    print("원본 이미지에서 얼굴을 찾지 못했습니다.")
else:
    face_orig = faces_orig[0]



    # 정렬 이미지 탐지
    embedding_aligned = get_embedding_from_aligned_image("../images/aligned_face1.png", ort_session)
    embedding_aligned2 = get_embedding_from_aligned_image("../images/aligned_face2.png", ort_session)
    embedding_aligned3 = get_embedding_from_aligned_image("../images/aligned_face3.png", ort_session)

    # 코사인 유사도 함수
    def cosine_similarity(vec1, vec2):
        vec1 = vec1.reshape(-1)
        vec2 = vec2.reshape(-1)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 임베딩 간 유사도 계산
    org_1_2 = cosine_similarity(emb_1, emb_2)
    org_1_3 = cosine_similarity(emb_1, emb_3)
    org_2_3 = cosine_similarity(emb_2, emb_3)

    print(f"Similarity between aligned_from_orig and aligned_from_orig2: {org_1_2:.6f}")
    print(f"Similarity between aligned_from_orig and aligned_from_orig3: {org_1_3:.6f}")
    print(f"Similarity between aligned_from_orig2 and aligned_from_orig3: {org_2_3:.6f}")


    # 임베딩 간 유사도 계산
    my_1_2 = cosine_similarity(embedding_aligned, embedding_aligned2)
    my_1_3 = cosine_similarity(embedding_aligned, embedding_aligned3)
    my_2_3 = cosine_similarity(embedding_aligned2, embedding_aligned3)

    print(f"Similarity between aligned_from_my1 and aligned_from_my2: {my_1_2:.6f}")
    print(f"Similarity between aligned_from_my1 and aligned_from_my3: {my_1_3:.6f}")
    print(f"Similarity between aligned_from_my2 and aligned_from_my3: {my_2_3:.6f}")

    # 임베딩 간 유사도 계산
    sim_1_2 = cosine_similarity(emb_1, embedding_aligned)
    sim_1_3 = cosine_similarity(emb_2, embedding_aligned2)
    sim_2_3 = cosine_similarity(emb_3, embedding_aligned3)

    print(f"Similarity between aligned_from_orig and aligned_from_my1: {sim_1_2:.6f}")
    print(f"Similarity between aligned_from_orig2 and aligned_from_my2: {sim_1_3:.6f}")
    print(f"Similarity between aligned_from_orig3 and aligned_from_my3: {sim_2_3:.6f}")
