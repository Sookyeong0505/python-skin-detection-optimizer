# 선정성 탐지 임계값 최적화 도구

AI 모델을 사용한 선정성 탐지에서 정탐과 오탐을 구분하는 최적의 임계값을 찾기 위한 데이터 수집 도구입니다.

## 주요 기능

- 🔍 **모델 선택**: 다양한 YOLO 모델 중 선택
- 📸 **이미지 분석**: 복수 이미지 파일 업로드 및 배치 분석
- 🎯 **분류 지원**: 정탐/오탐 분류 및 특징 기록
- 📊 **결과 시각화**: 실시간 분석 결과 및 통계 표시
- 💾 **데이터 저장**: 엑셀 형태로 결과 내보내기

## 설치 방법

### 1. 필요 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 실제 모델 사용을 위한 추가 패키지 (선택사항)

```bash
pip install opencv-python onnxruntime
```

## 사용 방법

### 기본 사용 (모의 데이터)

```bash
streamlit run sensitivity_detection_tool.py
```

### 실제 모델 사용

1. **모델 파일 준비**
   - `best_640n_0522.onnx` 등의 ONNX 모델 파일을 준비
   - 모델 파일을 프로젝트 폴더에 위치

2. **코드 수정**
   ```python
   # sensitivity_detection_tool.py의 analyze_image 함수를 수정
   from model_analyzer import analyze_with_real_models
   
   # 모의 분석 대신 실제 모델 사용
   analysis_result = analyze_with_real_models(
       model_path=f"./models/{selected_model}",
       image_files=[file],
       confidence_threshold=confidence_threshold
   )[0]
   ```

## 프로젝트 구조

```
├── sensitivity_detection_tool.py    # 메인 Streamlit 앱
├── model_analyzer.py               # 실제 모델 연동 코드
├── requirements.txt               # 필요 패키지 목록
├── README.md                     # 사용 가이드
└── models/                      # 모델 파일 저장 폴더
    ├── best_640n_0522.onnx
    └── ...
```

## 분석 결과 데이터

분석된 데이터는 다음과 같은 컬럼으로 구성됩니다:

### 기본 정보
- `id`: 고유 ID
- `analysis_date`: 분석 날짜
- `file_name`: 파일명
- `file_size_kb`: 파일 크기

### 모델 설정
- `model_name`: 사용된 모델명
- `input_size`: 입력 이미지 크기
- `confidence_threshold`: 신뢰도 임계값

### YOLO 탐지 결과
- `detected_objects_count`: 탐지된 객체 수
- `detected_classes`: 탐지된 클래스 목록
- `highest_confidence_score`: 최고 신뢰도 점수
- `exposed_breast_f_confidence`: 여성 가슴 신뢰도
- `exposed_breast_m_confidence`: 남성 가슴 신뢰도
- `exposed_buttocks_confidence`: 엉덩이 신뢰도
- `exposed_genitalia_f_confidence`: 여성 성기 신뢰도
- `exposed_genitalia_m_confidence`: 남성 성기 신뢰도

### 살색 분석 결과
- `total_skin_ratio`: 전체 살색 비율(%)
- `cr_channel_concentration`: Cr 채널 집중도
- `cb_channel_concentration`: Cb 채널 집중도
- `skin_threshold_25_result`: 임계값 25% 통과 여부
- `skin_threshold_40_result`: 임계값 40% 통과 여부

### 분류 및 평가
- `classification`: 정탐/오탐 분류
- `false_positive_features`: 오탐 특징 설명
- `analyst_notes`: 분석자 메모

### 성능 지표
- `processing_duration_sec`: 처리 소요 시간(초)

## 사용 단계

1. **모델 선택**: 드롭다운에서 분석할 YOLO 모델 선택
2. **임계값 설정**: 신뢰도 임계값 조정 (0.1 ~ 0.9)
3. **이미지 업로드**: 분석할 .webp, .jpg, .png 이미지 선택
4. **분류 설정**: 이미지가 정탐인지 오탐인지 선택
5. **특징 입력**: 오탐인 경우 오탐 특징 설명 입력
6. **분석 실행**: 버튼 클릭으로 배치 분석 수행
7. **결과 확인**: 분석 결과 및 통계 확인
8. **데이터 저장**: 엑셀 파일로 결과 다운로드

## 주의사항

- 이미지 형식: .webp, .jpg, .jpeg, .png 지원
- 메모리 사용량: 대용량 이미지나 많은 파일 처리 시 주의
- 모델 경로: 실제 모델 사용 시 정확한 경로 설정 필요

## 문제 해결

### 모델 로드 오류
- ONNX 모델 파일 경로 확인
- ONNX Runtime 설치 확인
- 모델 파일 손상 여부 확인

### 이미지 분석 오류
- 이미지 파일 형식 확인
- 파일 크기 및 해상도 확인
- OpenCV 설치 확인

### 성능 이슈
- 이미지 크기 조정 고려
- 배치 크기 줄이기
- GPU 사용 설정 (ONNX Runtime GPU 버전)

## 라이센스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 연락처

프로젝트 관련 문의사항이 있으시면 개발팀에 연락해주세요.