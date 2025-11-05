
### BlazeFace (Face Detector)
* https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/index?hl=ko#models
  * Get blaze_face_short_range.tflite (https://storage.googleapis.com/mediapipe-assets/blaze_face_short_range.tflite)
  * pip install numpy==1.26 tensorflow tf2onnx onnx
  * python -m tf2onnx.convert --tflite ./blaze_face_short_range.tflite --output ./face_detector.onnx --opset 17
    * Inputs:<br>
      input: [1, 128, 128, 3]    // 배치(단일이미지처리), 가로, 세로, 채널(RGB)
    * Outputs:<br>
      regressors: [1, 896, 16]    // 단일이미지, 앵커박스수, 16 얼굴좌표영역[4] + 얼굴키폰인트(x,y) 6쌍 [12]
      classificators: [1, 896, 1] // 단일이미지, 앵커박스수, 신뢰도
* https://huggingface.co/garavv/blazeface-onnx (blaze.onnx)
  
### Insightface  (Landmark + Recognition)
* buffalo_s (https://drive.google.com/file/d/1pKIusApEfoHKDjeBTXYB3yOQ0EtTonNE/view?usp=sharing)
  * 1k3d68.onnx (3D Face Landmark)
  * 2d106det.onnx (2D Face Landmark)
  * det_500m.onnx (Face Detection + 5 Landmark)
    * Inputs:<br>
      input.1: [1, 3, 0, 0]   // 배치, 채널, 높이, 넓이 (일반적으로 정사각형)
    * Outputs:<br>
      443: [12800, 1]  // 큰 특징맵(작은얼굴): 12800개 앵커 박스 각각의 얼굴 존재 확률 (320x320 => 3200) 
      468: [3200, 1]   // 중간 특징맵(중간얼굴): 3200개 앵커 박스의 얼굴 확률    (320x320 =>  800)
      493: [800, 1]  // 작은 특징맵(큰얼굴): 800개 앵커 박스의 얼굴확률     (320x320 => 200)
      446: [12800, 4]  // 12800개 앵커박스의 바운딩 박스 좌표 회귀값(x,y,w,h)  
      471: [3200, 4]  // 3200개 앵커박스의 바운딩 박스 회귀값    
      496: [800, 4]  //800개 앵커 박스의 바운딩 박스 회귀값  
      449: [12800, 10]  // 12800개 앵커 박스의 5개 얼굴 랜드마크 좌표  
      474: [3200, 10] // 3200개 앵커 박스의 랜드박스 좌표   
      499: [800, 10]  // 800개 앵커 박스의 5개 랜드마크 좌표  
    * 5개 랜드마크:<br> 
      * 왼쪽 눈
      * 오른쪽 눈
      * 코
      * 왼쪽 입꼬리
      * 오른쪽 입꼬리
  * genderage.onnx (Gender, Age)
  * w600k-mbf.onnx (Face Recognition)
    * Inputs:<br>
      input.1: [0, 3, 112, 112]
    * Outputs:<br>
      516: [1, 512]

###[ TEST DATA]
* lfw: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?resource=download
  * lfw 폴더 생성하여 해당 자료 추가
  * cargo run --bin lfw
* celeba-hq: https://www.kaggle.com/datasets/lamsimon/celebahq
  * celeba 폴더 생성하여 해당 자료 추가
  * cargo run --bin celeba