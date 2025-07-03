# model_analyzer.py - 실제 모델 연동을 위한 별도 모듈
import os

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

class YOLOAnalyzer:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.session = ort.InferenceSession(model_path)
        self.input_size = 640  # 모델에 따라 조정

        # 클래스 라벨 정의
        self.class_names = [
            'EXPOSED_BREAST_F',
            'EXPOSED_BREAST_M',
            'EXPOSED_BUTTOCKS',
            'EXPOSED_GENITALIA_F',
            'EXPOSED_GENITALIA_M'
        ]

    def preprocess_image(self, image):
        """이미지 전처리"""
        # PIL Image를 numpy array로 변환
        if isinstance(image, Image.Image):
            image = np.array(image)

        # BGR to RGB 변환 (OpenCV 사용 시)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 리사이즈
        image_resized = cv2.resize(image, (self.input_size, self.input_size))

        # 정규화 및 차원 변경 (NCHW 형식)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_transposed, axis=0)

        return image_batch

    def analyze(self, image):
        """YOLO 모델로 이미지 분석"""
        try:
            # 이미지 전처리
            input_data = self.preprocess_image(image)

            # 모델 추론
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})

            # 결과 파싱
            results = self.parse_outputs(outputs[0])

            return results

        except Exception as e:
            # 에러 발생 시 기본값 반환
            print(f"YOLO 분석 중 오류 발생: {e}")
            return self.get_default_results()

    def parse_outputs(self, outputs):
        """YOLO 출력 결과 파싱"""
        detected_objects = []
        confidences = {cls.lower() + '_confidence': 0.0 for cls in self.class_names}

        # YOLO 출력 파싱 (모델에 따라 조정 필요)
        # 이 부분은 실제 모델의 출력 형식에 맞게 구현해야 합니다

        # 예시 구현 (실제로는 모델 출력에 맞게 수정)
        for detection in outputs:
            if len(detection) >= 6:  # [x, y, w, h, confidence, class_id]
                confidence = detection[4]
                class_id = int(detection[5])

                if confidence >= self.confidence_threshold and class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    detected_objects.append(class_name)
                    confidences[class_name.lower() + '_confidence'] = max(
                        confidences[class_name.lower() + '_confidence'],
                        confidence
                    )

        # 결과 정리
        highest_confidence = max(confidences.values())
        highest_class = ""
        if highest_confidence > 0:
            for key, value in confidences.items():
                if value == highest_confidence:
                    highest_class = key.replace('_confidence', '')
                    break

        return {
            'detected_objects_count': len(detected_objects),
            'detected_classes': ', '.join(list(set(detected_objects))),
            'highest_confidence_class': highest_class,
            'highest_confidence_score': round(highest_confidence, 3),
            **{k: round(v, 3) for k, v in confidences.items()}
        }

    def get_default_results(self):
        """기본 결과값 반환 (모델 로드 실패 시)"""
        confidences = {cls.lower() + '_confidence': 0.0 for cls in self.class_names}

        return {
            'detected_objects_count': 0,
            'detected_classes': '',
            'highest_confidence_class': '',
            'highest_confidence_score': 0.0,
            **confidences
        }

class SkinColorAnalyzer:
    def __init__(self):
        pass

    def analyze(self, image):
        """살색 분석 수행"""
        try:
            # PIL Image를 numpy array로 변환
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            # RGB to YCrCb 변환
            ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCR_CB)

            # 살색 범위 정의 (YCrCb 색공간에서)
            # 이 값들은 실제 연구나 실험을 통해 조정해야 합니다
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)

            # 살색 마스크 생성
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

            # 살색 픽셀 비율 계산
            total_pixels = image_np.shape[0] * image_np.shape[1]
            skin_pixels = np.sum(skin_mask > 0)
            skin_ratio = (skin_pixels / total_pixels) * 100

            # Cr, Cb 채널 분석
            cr_channel = ycrcb[:, :, 1]
            cb_channel = ycrcb[:, :, 2]

            # 살색 영역에서의 Cr, Cb 집중도 계산
            skin_cr = cr_channel[skin_mask > 0]
            skin_cb = cb_channel[skin_mask > 0]

            cr_concentration = np.mean(skin_cr) if len(skin_cr) > 0 else 0
            cb_concentration = np.mean(skin_cb) if len(skin_cb) > 0 else 0

            # 정규화된 비율
            cr_ratio = cr_concentration / 255.0 if cr_concentration > 0 else 0
            cb_ratio = cb_concentration / 255.0 if cb_concentration > 0 else 0

            return {
                'total_skin_ratio': round(skin_ratio, 2),
                'cr_channel_concentration': round(cr_concentration, 2),
                'cb_channel_concentration': round(cb_concentration, 2),
                'cr_skin_pixel_ratio': round(cr_ratio, 3),
                'cb_skin_pixel_ratio': round(cb_ratio, 3),
                'skin_threshold_25_result': skin_ratio >= 25.0,
                'skin_threshold_40_result': skin_ratio >= 40.0
            }

        except Exception as e:
            print(f"살색 분석 중 오류 발생: {e}")
            return self.get_default_results()

    def get_default_results(self):
        """기본 결과값 반환 (분석 실패 시)"""
        return {
            'total_skin_ratio': 0.0,
            'cr_channel_concentration': 0.0,
            'cb_channel_concentration': 0.0,
            'cr_skin_pixel_ratio': 0.0,
            'cb_skin_pixel_ratio': 0.0,
            'skin_threshold_25_result': False,
            'skin_threshold_40_result': False
        }

def analyze_image_with_models(model_path, image_file, confidence_threshold):
    """실제 모델을 사용한 이미지 분석"""
    import time

    start_time = time.time()

    # 이미지 로드
    image = Image.open(image_file)

    # YOLO 분석
    yolo_analyzer = YOLOAnalyzer(model_path, confidence_threshold)
    yolo_results = yolo_analyzer.analyze(image)

    # 살색 분석
    skin_analyzer = SkinColorAnalyzer()
    skin_results = skin_analyzer.analyze(image)

    # 처리 시간 계산
    processing_time = time.time() - start_time

    # 파일 정보
    file_info = {
        'file_name': image_file.name,
        'file_size_kb': round(len(image_file.getvalue()) / 1024, 2)
    }

    # 모델 설정
    model_info = {
        'model_name': os.path.basename(model_path),
        'input_size': yolo_analyzer.input_size,
        'confidence_threshold': confidence_threshold
    }

    # 성능 지표
    performance = {
        'processing_start_time': round(start_time * 1000, 1),
        'processing_duration_sec': round(processing_time, 3)
    }

    return {
        **file_info,
        **model_info,
        **yolo_results,
        **skin_results,
        **performance
    }

# 실제 모델 사용을 위한 수정된 메인 함수
def analyze_with_real_models(model_path, image_files, confidence_threshold):
    """실제 모델을 사용한 배치 분석"""
    results = []

    for image_file in image_files:
        try:
            result = analyze_image_with_models(model_path, image_file, confidence_threshold)
            results.append(result)
        except Exception as e:
            print(f"이미지 {image_file.name} 분석 중 오류: {e}")
            # 오류 발생 시 기본값으로 처리
            continue

    return results