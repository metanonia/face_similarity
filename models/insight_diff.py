from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(3320,320))

# 원본 이미지
img_orig = cv2.imread("../models/cropped_face.jpg")
faces_orig = app.get(img_orig)
embedding_orig = faces_orig[0].embedding

# Aligned 이미지
img_aligned = cv2.imread("../models/aligned_face_112.png")
faces_aligned = app.get(img_aligned)
embedding_aligned = faces_aligned[0].embedding

# Cosine similarity
import numpy as np
cos_sim = np.dot(embedding_orig, embedding_aligned) / (np.linalg.norm(embedding_orig) * np.linalg.norm(embedding_aligned))
print("Cosine similarity:", cos_sim)
