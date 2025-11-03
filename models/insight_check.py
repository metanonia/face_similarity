import insightface
import cv2
import numpy as np

# 모델 로드
app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(320, 320))  # 320x320 입력 사용

# 이미지 로드
img = cv2.imread('../images/cropped_face.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 얼굴 감지
faces = app.get(img_rgb)

if len(faces) > 0:
    face = faces[0]

    print("="*70)
    print("FACE DETECTION DETAILS")
    print("="*70)

    # 바운딩 박스
    print("\n1. Bounding Box:")
    print(f"   Raw bbox: {face.bbox}")
    x1, y1, x2, y2 = face.bbox
    print(f"   Format: [x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f}]")
    print(f"   Width: {x2-x1:.4f}, Height: {y2-y1:.4f}")
    print(f"   Confidence: {face.det_score:.6f}")

    # Landmarks
    print("\n2. Landmarks (5 points):")
    print(f"   Shape: {face.kps.shape}")
    for j, kp in enumerate(face.kps):
        print(f"   Landmark {j}: x={kp[0]:.4f}, y={kp[1]:.4f}")

    # 크롭된 얼굴 확인
    print("\n3. Cropped Face:")
    x1_i, y1_i, x2_i, y2_i = map(int, face.bbox)
    cropped = img[y1_i:y2_i, x1_i:x2_i]
    print(f"   Size: {cropped.shape}")
    print(f"   Bbox (pixels): x1={x1_i}, y1={y1_i}, x2={x2_i}, y2={y2_i}")

    # 얼굴 인식 (embedding)
    print("\n4. Face Embedding:")
    print(f"   Embedding shape: {face.embedding.shape}")
    print(f"   First 5 values: {face.embedding[:5]}")

    print("\n" + "="*70)