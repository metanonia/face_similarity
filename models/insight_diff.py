from insightface.app import FaceAnalysis
import cv2
import numpy as np
import onnxruntime as ort
from insightface.utils import face_align
from pathlib import Path

# ArcFace 임베딩 모델 ONNX 경로
arcface_model_path = "w600k_mbf.onnx"
ort_session = ort.InferenceSession(arcface_model_path)

def preprocess_aligned_face_already_112(img):
    """
    이미 112x112로 정렬된 얼굴에 대해 전처리 수행
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

def get_embedding_from_aligned_img(img, ort_session):
    input_tensor = preprocess_aligned_face_already_112(img)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_tensor})
    embedding = outputs[0][0]
    return embedding

# 코사인 유사도 함수
def cosine_similarity(vec1, vec2):
    vec1 = vec1.reshape(-1)
    vec2 = vec2.reshape(-1)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# InsightFace 준비
print("=" * 80)
print("얼굴 임베딩 유사도 비교 (ONNX vs InsightFace)")
print("=" * 80)

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(320, 320))

# ============================================================================
# 1. ONNX 모델을 사용한 임베딩 계산 (직접 저장된 정렬 이미지 사용)
# ============================================================================
print("\n" + "=" * 80)
print("1. ONNX 모델을 사용한 임베딩 계산")
print("=" * 80)

aligned_image_paths = [
    "../images/aligned_face01_0_0.png",
    "../images/aligned_face02_0_0.png",
    "../images/aligned_face03_0_0.png",
    "../images/aligned_face04_0_0.png",
    "../images/aligned_face05_0_0.png",
    "../images/aligned_face06_0_0.png",
    "../images/aligned_face07_0_0.png",
]

onnx_embeddings = {}
print("\n✅ ONNX 임베딩 추출 중...")

for img_path in aligned_image_paths:
    try:
        embedding = get_embedding_from_aligned_image(img_path, ort_session)
        face_name = Path(img_path).stem  # 파일명 추출
        onnx_embeddings[face_name] = embedding
        print(f"   {face_name}: 완료")
    except Exception as e:
        print(f"   ❌ {img_path}: {e}")

# ============================================================================
# 2. InsightFace 라이브러리를 사용한 임베딩 계산 (원본 이미지 사용)
# ============================================================================
print("\n" + "=" * 80)
print("2. InsightFace 라이브러리를 사용한 임베딩 계산")
print("=" * 80)

insightface_embeddings = {}
print("\n✅ InsightFace 임베딩 추출 중...")

for i in range(1, 8):  # face01.jpg ~ face06.jpg
    img_path = f"../images/face{i:02d}.jpg"

    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"   ⚠️ {img_path}: 파일을 찾을 수 없습니다.")
            continue

        # InsightFace로 얼굴 감지 및 임베딩
        faces = app.get(img)

        if len(faces) > 0:
            # 첫 번째 얼굴의 임베딩 사용
            embedding = faces[0].embedding
            face_name = f"face{i:02d}"
            insightface_embeddings[face_name] = embedding
            print(f"   ✅ {face_name}: 완료 (감지된 얼굴: {len(faces)}개)")
        else:
            print(f"   ⚠️ {img_path}: 얼굴이 감지되지 않았습니다.")

    except Exception as e:
        print(f"   ❌ {img_path}: {e}")

# ============================================================================
# 3. 유사도 행렬 계산 (ONNX)
# ============================================================================
print("\n" + "=" * 80)
print("3. ONNX 모델 유사도 분석")
print("=" * 80)

if len(onnx_embeddings) > 1:
    faces_onnx = list(onnx_embeddings.keys())
    similarity_matrix_onnx = np.zeros((len(faces_onnx), len(faces_onnx)))

    print("\n✅ ONNX 코사인 유사도:")
    for i, face1 in enumerate(faces_onnx):
        for j, face2 in enumerate(faces_onnx):
            if i <= j:
                sim = cosine_similarity(onnx_embeddings[face1], onnx_embeddings[face2])
                similarity_matrix_onnx[i][j] = sim
                similarity_matrix_onnx[j][i] = sim

                if i != j:
                    print(f"   {face1} vs {face2}: {sim:.6f}")
                else:
                    print(f"   {face1} vs {face2}: {sim:.6f} (자기자신)")

    # ONNX 유사도 행렬
    print("\n✅ ONNX 유사도 행렬:")
    print("      ", end="")
    for face in faces_onnx:
        print(f"{face:15}", end="")
    print()

    for i, face1 in enumerate(faces_onnx):
        print(f"{face1}", end=" ")
        for j in range(len(faces_onnx)):
            print(f"{similarity_matrix_onnx[i][j]:15.6f}", end="")
        print()

# ============================================================================
# 4. 유사도 행렬 계산 (InsightFace)
# ============================================================================
print("\n" + "=" * 80)
print("4. InsightFace 라이브러리 유사도 분석")
print("=" * 80)

if len(insightface_embeddings) > 1:
    faces_insightface = sorted(list(insightface_embeddings.keys()))
    similarity_matrix_insightface = np.zeros((len(faces_insightface), len(faces_insightface)))

    print("\n✅ InsightFace 코사인 유사도:")
    for i, face1 in enumerate(faces_insightface):
        for j, face2 in enumerate(faces_insightface):
            if i <= j:
                sim = cosine_similarity(insightface_embeddings[face1], insightface_embeddings[face2])
                similarity_matrix_insightface[i][j] = sim
                similarity_matrix_insightface[j][i] = sim

                if i != j:
                    print(f"   {face1} vs {face2}: {sim:.6f}")
                else:
                    print(f"   {face1} vs {face2}: {sim:.6f} (자기자신)")

    # InsightFace 유사도 행렬
    print("\n✅ InsightFace 유사도 행렬:")
    print("      ", end="")
    for face in faces_insightface:
        print(f"{face:15}", end="")
    print()

    for i, face1 in enumerate(faces_insightface):
        print(f"{face1}", end=" ")
        for j in range(len(faces_insightface)):
            print(f"{similarity_matrix_insightface[i][j]:15.6f}", end="")
        print()

# ============================================================================
# 5. 비교 분석
# ============================================================================
print("\n" + "=" * 80)
print("5. 최종 분석")
print("=" * 80)

if len(onnx_embeddings) > 1:
    print("\n✅ ONNX 분석:")
    max_sim_onnx = -1
    max_pair_onnx = None
    for i in range(len(faces_onnx)):
        for j in range(i + 1, len(faces_onnx)):
            if similarity_matrix_onnx[i][j] > max_sim_onnx:
                max_sim_onnx = similarity_matrix_onnx[i][j]
                max_pair_onnx = (faces_onnx[i], faces_onnx[j])

    print(f"   가장 유사한 쌍: {max_pair_onnx[0]} vs {max_pair_onnx[1]} ({max_sim_onnx:.6f})")

    avg_sims_onnx = []
    for i, face in enumerate(faces_onnx):
        other_sims = [similarity_matrix_onnx[i][j] for j in range(len(faces_onnx)) if i != j]
        avg_sim = np.mean(other_sims)
        avg_sims_onnx.append(avg_sim)
        print(f"   {face} 평균 유사도: {avg_sim:.6f}")

if len(insightface_embeddings) > 1:
    print("\n✅ InsightFace 분석:")
    max_sim_if = -1
    max_pair_if = None
    for i in range(len(faces_insightface)):
        for j in range(i + 1, len(faces_insightface)):
            if similarity_matrix_insightface[i][j] > max_sim_if:
                max_sim_if = similarity_matrix_insightface[i][j]
                max_pair_if = (faces_insightface[i], faces_insightface[j])

    print(f"   가장 유사한 쌍: {max_pair_if[0]} vs {max_pair_if[1]} ({max_sim_if:.6f})")

    for i, face in enumerate(faces_insightface):
        other_sims = [similarity_matrix_insightface[i][j] for j in range(len(faces_insightface)) if i != j]
        avg_sim = np.mean(other_sims)
        print(f"   {face} 평균 유사도: {avg_sim:.6f}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
