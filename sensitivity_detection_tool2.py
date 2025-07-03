import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
import os
from PIL import Image, ImageDraw
import io
import base64
import cv2

# 페이지 설정
st.set_page_config(
    page_title="선정성 탐지 임계값 최적화 도구",
    page_icon="🔍",
    layout="wide"
)

# 세션 상태 초기화
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

if 'image_classifications' not in st.session_state:
    st.session_state.image_classifications = {}

if 'image_notes' not in st.session_state:
    st.session_state.image_notes = {}

if 'final_results' not in st.session_state:
    st.session_state.final_results = []

# 한글 클래스명 매핑
CLASS_NAMES_KR = {
    'EXPOSED_BREAST_F': '여성_가슴_노출',
    'EXPOSED_BREAST_M': '남성_가슴_노출',
    'EXPOSED_BUTTOCKS': '엉덩이_노출',
    'EXPOSED_GENITALIA_F': '여성_성기_노출',
    'EXPOSED_GENITALIA_M': '남성_성기_노출'
}

# 엑셀 컬럼 헤더 한글 매핑
COLUMN_NAMES_KR = {
    'id': 'ID',
    'analysis_date': '분석_날짜',
    'file_name': '파일명',
    'file_path': '파일_경로',
    'file_size_kb': '파일_크기_KB',
    'model_name': '모델명',
    'input_size': '입력_이미지_크기',
    'confidence_threshold': '신뢰도_임계값',
    'detected_objects_count': '탐지된_객체_수',
    'detected_classes': '탐지된_클래스_목록',
    'highest_confidence_class': '최고_신뢰도_클래스',
    'highest_confidence_score': '최고_신뢰도_점수',
    'exposed_breast_f_confidence': '여성_가슴_신뢰도',
    'exposed_breast_m_confidence': '남성_가슴_신뢰도',
    'exposed_buttocks_confidence': '엉덩이_신뢰도',
    'exposed_genitalia_f_confidence': '여성_성기_신뢰도',
    'exposed_genitalia_m_confidence': '남성_성기_신뢰도',
    'total_skin_ratio': '전체_살색_비율_퍼센트',
    'cr_channel_concentration': 'Cr_채널_집중도_퍼센트',
    'cb_channel_concentration': 'Cb_채널_집중도_퍼센트',
    'cr_skin_pixel_ratio': 'Cr_살색_픽셀_비율',
    'cb_skin_pixel_ratio': 'Cb_살색_픽셀_비율',
    'skin_threshold_25_result': '임계값_25퍼센트_통과_여부',
    'skin_threshold_40_result': '임계값_40퍼센트_통과_여부',
    'classification': '정탐_오탐_분류',
    'false_positive_features': '오탐_특징_설명',
    'analyst_notes': '분석자_메모',
    'processing_start_time': '처리_시작_시간_ms',
    'total_processing_time': '총_처리_시간_ms',
    'processing_duration_sec': '처리_소요_시간_초',
    'skin_confidence_combined': '살색_객체탐지_종합점수',
    'risk_level': '위험도_등급',
    'filter_recommendation': '필터_권장사항'
}

def generate_mock_yolo_results():
    """YOLO 모델 분석 결과 모의 데이터 생성"""
    classes = ['EXPOSED_BREAST_F', 'EXPOSED_BREAST_M', 'EXPOSED_BUTTOCKS',
               'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M']

    detected_count = random.randint(0, 3)
    detected_classes = random.sample(classes, min(detected_count, len(classes))) if detected_count > 0 else []

    confidences = {}
    for cls in classes:
        if cls in detected_classes:
            confidences[cls.lower().replace('exposed_', '') + '_confidence'] = round(random.uniform(0.3, 0.95), 3)
        else:
            confidences[cls.lower().replace('exposed_', '') + '_confidence'] = 0.0

    highest_confidence = max(confidences.values()) if confidences else 0.0
    highest_class = ""
    if highest_confidence > 0:
        for key, value in confidences.items():
            if value == highest_confidence:
                highest_class = key.replace('_confidence', '')
                break

    return {
        'detected_objects_count': detected_count,
        'detected_classes': ', '.join([CLASS_NAMES_KR.get(cls, cls) for cls in detected_classes]),
        'highest_confidence_class': CLASS_NAMES_KR.get('EXPOSED_' + highest_class.upper(), highest_class),
        'highest_confidence_score': highest_confidence,
        **confidences
    }

def generate_mock_skin_analysis():
    """살색 분석 결과 모의 데이터 생성"""
    total_skin_ratio = round(random.uniform(5.0, 99.0), 2)

    return {
        'total_skin_ratio': total_skin_ratio,
        'cr_channel_concentration': round(random.uniform(20.0, 40.0), 2),
        'cb_channel_concentration': round(random.uniform(40.0, 60.0), 2),
        'cr_skin_pixel_ratio': round(random.uniform(0.2, 0.999), 3),
        'cb_skin_pixel_ratio': round(random.uniform(0.4, 0.998), 3),
        'skin_threshold_25_result': total_skin_ratio >= 25,
        'skin_threshold_40_result': total_skin_ratio >= 40
    }

def create_mock_detection_image(original_image):
    """모의 탐지 결과 이미지 생성"""
    img_copy = original_image.copy()
    draw = ImageDraw.Draw(img_copy)

    # 임의의 바운딩 박스 그리기
    width, height = img_copy.size
    x1 = random.randint(int(width*0.2), int(width*0.5))
    y1 = random.randint(int(height*0.2), int(height*0.5))
    x2 = x1 + random.randint(int(width*0.2), int(width*0.4))
    y2 = y1 + random.randint(int(height*0.2), int(height*0.4))

    # 바운딩 박스와 라벨 그리기
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

    classes = list(CLASS_NAMES_KR.keys())
    detected_class = random.choice(classes)
    confidence = round(random.uniform(0.3, 0.95), 2)
    label = f"{CLASS_NAMES_KR[detected_class]} ({confidence}%)"

    draw.rectangle([x1, y1-25, x1+200, y1], fill="lime")
    draw.text((x1+5, y1-20), label, fill="black")

    return img_copy, (x1, y1, x2, y2)

def create_cropped_region(original_image, bbox):
    """최고 스코어 영역 크롭"""
    x1, y1, x2, y2 = bbox
    return original_image.crop((x1, y1, x2, y2))

def create_skin_analysis_image(original_image):
    """살색 분석 결과 이미지 생성 (YCrCb)"""
    # PIL을 numpy로 변환
    img_array = np.array(original_image)

    # RGB to YCrCb 변환을 모의
    # 실제로는 cv2.cvtColor를 사용하지만 여기서는 단순화
    height, width = img_array.shape[:2]

    # 빨간색 톤의 이미지 생성 (살색 분석 결과처럼)
    skin_mask = np.full((height, width, 3), [255, 100, 100], dtype=np.uint8)

    # 일부 영역을 더 진한 빨간색으로
    mask_region = np.random.rand(height, width) > 0.7
    skin_mask[mask_region] = [200, 50, 50]

    return Image.fromarray(skin_mask)

def analyze_image(model_name, image_file, confidence_threshold):
    """이미지 분석 수행"""
    # 파일 정보
    file_info = {
        'file_name': image_file.name,
        'file_size_kb': round(len(image_file.getvalue()) / 1024, 2)
    }

    # 모델 설정
    model_info = {
        'model_name': model_name,
        'input_size': 640 if 'best_640n' in model_name else 320,
        'confidence_threshold': confidence_threshold
    }

    # YOLO 분석 결과
    yolo_results = generate_mock_yolo_results()

    # 살색 분석 결과
    skin_results = generate_mock_skin_analysis()

    # 성능 지표
    processing_time = round(random.uniform(0.1, 0.5), 3)
    performance = {
        'processing_start_time': round(datetime.now().timestamp() * 1000, 1),
        'processing_duration_sec': processing_time
    }

    return {
        **file_info,
        **model_info,
        **yolo_results,
        **skin_results,
        **performance
    }

def main():
    st.title("🔍 선정성 탐지 임계값 최적화 도구")
    st.markdown("---")

    # 사이드바 - 현재 결과 통계
    st.sidebar.header("📊 현재 데이터 통계")
    if st.session_state.final_results:
        df = pd.DataFrame(st.session_state.final_results)
        total_count = len(df)
        positive_count = len(df[df['classification'] == '정탐'])
        negative_count = len(df[df['classification'] == '오탐'])

        st.sidebar.metric("총 분석 이미지", total_count)
        st.sidebar.metric("정탐 이미지", positive_count)
        st.sidebar.metric("오탐 이미지", negative_count)

        if total_count > 0:
            st.sidebar.metric("정탐률", f"{positive_count/total_count*100:.1f}%")
    else:
        st.sidebar.info("아직 분석된 데이터가 없습니다.")

    # 메인 인터페이스
    st.header("1. 모델 및 설정")

    col1, col2, col3 = st.columns(3)

    with col1:
        # 모델 선택
        model_options = [
            "best_640n_0522.onnx",
            "best_320n_0415.onnx",
            "experimental_model_v2.onnx"
        ]
        selected_model = st.selectbox("분석할 모델 선택", model_options)

    with col2:
        # 신뢰도 임계값
        confidence_threshold = st.slider("신뢰도 임계값", 0.1, 0.9, 0.5, 0.1)

    with col3:
        # 이미지 크기 표시
        input_size = 640 if 'best_640n' in selected_model else 320
        st.metric("입력 이미지 크기", f"{input_size}×{input_size}")

    st.markdown("---")

    # 이미지 업로드
    st.header("2. 이미지 업로드")

    uploaded_files = st.file_uploader(
        "분석할 이미지 선택 (.webp, .jpg, .png)",
        type=['webp', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="image_uploader"
    )

    if uploaded_files:
        if st.button("📋 이미지 목록 업데이트", type="primary"):
            st.session_state.uploaded_images = []
            st.session_state.analysis_results = {}

            # 업로드된 이미지들을 세션에 저장하고 분석
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                progress_bar.progress((i + 1) / len(uploaded_files))

                # 이미지 정보 저장
                image_data = {
                    'file': file,
                    'name': file.name,
                    'size': len(file.getvalue())
                }
                st.session_state.uploaded_images.append(image_data)

                # 이미지 분석 수행
                analysis_result = analyze_image(selected_model, file, confidence_threshold)
                st.session_state.analysis_results[file.name] = analysis_result

            progress_bar.empty()
            st.success(f"✅ {len(uploaded_files)}개 이미지가 업로드되고 분석되었습니다!")

    # 이미지 목록 표시
    if st.session_state.uploaded_images:
        st.markdown("---")
        st.header("3. 이미지 목록 및 분석 결과")

        # 게시판 형태로 이미지 목록 표시
        st.subheader("📋 분석된 이미지 목록")

        # 테이블 데이터 준비
        table_data = []
        for idx, image_data in enumerate(st.session_state.uploaded_images):
            if image_data['name'] in st.session_state.analysis_results:
                result = st.session_state.analysis_results[image_data['name']]

                # 저장 여부 확인
                is_saved = any(r['file_name'] == image_data['name'] for r in st.session_state.final_results)
                save_status = "✅ 저장됨" if is_saved else "❌ 미저장"

                # 검출된 객체명 (한글)
                detected_objects = result['detected_classes'] if result['detected_classes'] else "없음"

                table_data.append({
                    '번호': idx + 1,
                    '파일명': image_data['name'],
                    '검출된_객체': detected_objects,
                    '최고_신뢰도': f"{result['highest_confidence_score']:.3f}",
                    '살색_비율': f"{result['total_skin_ratio']:.1f}%",
                    '저장_여부': save_status,
                    '상세_분석': f"detail_{idx}"
                })

        # 테이블로 표시
        if table_data:
            df_display = pd.DataFrame(table_data)

            # 테이블 표시 (상세분석 버튼 제외)
            display_df = df_display[['번호', '파일명', '검출된_객체', '최고_신뢰도', '살색_비율', '저장_여부']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("**💡 상세 분석을 원하는 이미지를 선택하세요:**")

            # 각 이미지별 상세 분석 버튼을 행으로 표시
            for idx, row in enumerate(table_data):
                col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 3, 2, 1, 1, 1, 1.5])

                with col1:
                    st.write(f"{row['번호']}")
                with col2:
                    st.write(f"{row['파일명']}")
                with col3:
                    st.write(f"{row['검출된_객체']}")
                with col4:
                    st.write(f"{row['최고_신뢰도']}")
                with col5:
                    st.write(f"{row['살색_비율']}")
                with col6:
                    st.write(f"{row['저장_여부']}")
                with col7:
                    if st.button(f"상세 분석", key=f"detail_btn_{idx}"):
                        st.session_state.selected_image = st.session_state.uploaded_images[idx]['name']

    # 선택된 이미지 상세 분석
    if hasattr(st.session_state, 'selected_image') and st.session_state.selected_image:
        st.markdown("---")
        st.header(f"4. 상세 분석 결과: {st.session_state.selected_image}")

        # 선택된 이미지 찾기
        selected_file = None
        for img_data in st.session_state.uploaded_images:
            if img_data['name'] == st.session_state.selected_image:
                selected_file = img_data['file']
                break

        if selected_file and st.session_state.selected_image in st.session_state.analysis_results:
            original_image = Image.open(selected_file)
            result = st.session_state.analysis_results[st.session_state.selected_image]

            # 탐지 결과 이미지들 생성
            detection_image, bbox = create_mock_detection_image(original_image)
            cropped_region = create_cropped_region(original_image, bbox)
            skin_analysis_image = create_skin_analysis_image(original_image)

            # 이미지 표시
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("원본 이미지")
                st.image(original_image, use_column_width=True)

            with col2:
                st.subheader("검출 결과")
                st.image(detection_image, use_column_width=True)

            with col3:
                st.subheader("최고 스코어 영역")
                st.image(cropped_region, use_column_width=True)

            # 분석 결과 상세 정보
            st.subheader("📊 탐지 결과 상세")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("탐지된 객체 수", result['detected_objects_count'])
            with col2:
                st.metric("최고 신뢰도", f"{result['highest_confidence_score']:.3f}")
            with col3:
                st.metric("최고 신뢰도 클래스", result['highest_confidence_class'])
            with col4:
                st.metric("살색 비율", f"{result['total_skin_ratio']:.1f}%")

            # 살색 분석 결과
            st.subheader("🎨 살색 분석 결과 (YCrCb + 히스토그램)")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(skin_analysis_image, caption="YCrCb 색공간 살색 분석", use_column_width=True)

            with col2:
                st.write("**YCrCb 색공간 분석**")
                st.write(f"• 살색 비율: {result['total_skin_ratio']:.2f}%")
                st.write(f"• Cr 채널 집중도: {result['cr_channel_concentration']:.2f}%")
                st.write(f"• Cb 채널 집중도: {result['cb_channel_concentration']:.2f}%")
                st.write(f"• Cr 살색 픽셀: {result['cr_skin_pixel_ratio']:.3f}")
                st.write(f"• Cb 살색 픽셀: {result['cb_skin_pixel_ratio']:.3f}")

                st.write("**임계값 통과 여부**")
                threshold_25 = "✅ 통과" if result['skin_threshold_25_result'] else "❌ 미통과"
                threshold_40 = "✅ 통과" if result['skin_threshold_40_result'] else "❌ 미통과"
                st.write(f"• 임계값 25%: {threshold_25}")
                st.write(f"• 임계값 40%: {threshold_40}")

                # 최종 판정
                if result['total_skin_ratio'] >= 40:
                    st.error("🚨 살색 감지됨")
                elif result['total_skin_ratio'] >= 25:
                    st.warning("⚠️ 살색 의심")
                else:
                    st.success("✅ 정상")

            # 개별 이미지 분류 및 메모
            st.markdown("---")
            st.subheader("5. 분류 및 메모")

            col1, col2 = st.columns(2)

            with col1:
                # 정탐/오탐 선택
                classification_key = f"classification_{st.session_state.selected_image}"
                classification = st.radio(
                    "이미지 분류",
                    ["정탐", "오탐"],
                    key=classification_key,
                    help="이 이미지가 올바르게 탐지되었는지(정탐) 잘못 탐지되었는지(오탐) 선택하세요."
                )
                st.session_state.image_classifications[st.session_state.selected_image] = classification

                # 오탐 특징 (오탐인 경우에만)
                if classification == "오탐":
                    false_positive_features = st.text_area(
                        "오탐 특징 설명",
                        key=f"features_{st.session_state.selected_image}",
                        placeholder="예: 음식 이미지, 도넛 형태, 살색과 유사한 색상 등"
                    )
                else:
                    false_positive_features = ""

            with col2:
                # 분석자 메모
                notes_key = f"notes_{st.session_state.selected_image}"
                notes = st.text_area(
                    "분석자 메모",
                    key=notes_key,
                    placeholder="추가적인 관찰 사항이나 메모를 입력하세요",
                    height=150
                )
                st.session_state.image_notes[st.session_state.selected_image] = notes

            # 저장 버튼
            if st.button("💾 이 이미지 분석 결과 저장", type="primary"):
                # 결과에 분류 정보 추가
                final_result = result.copy()
                final_result.update({
                    'id': len(st.session_state.final_results) + 1,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'classification': classification,
                    'false_positive_features': false_positive_features if classification == "오탐" else "",
                    'analyst_notes': notes
                })

                # 기존 결과에서 같은 파일 제거 후 추가
                st.session_state.final_results = [r for r in st.session_state.final_results
                                                  if r['file_name'] != st.session_state.selected_image]
                st.session_state.final_results.append(final_result)

                st.success(f"✅ {st.session_state.selected_image} 분석 결과가 저장되었습니다!")

    # 전체 데이터 관리
    st.markdown("---")
    st.header("6. 💾 데이터 관리")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 전체 결과 보기"):
            if st.session_state.final_results:
                df_all = pd.DataFrame(st.session_state.final_results)
                display_columns = [
                    'file_name', 'classification', 'highest_confidence_score',
                    'total_skin_ratio', 'detected_objects_count', 'analyst_notes'
                ]
                st.dataframe(df_all[display_columns], use_container_width=True)
            else:
                st.info("저장된 결과가 없습니다.")

    with col2:
        if st.button("🗑️ 모든 데이터 삭제"):
            st.session_state.uploaded_images = []
            st.session_state.analysis_results = {}
            st.session_state.image_classifications = {}
            st.session_state.image_notes = {}
            st.session_state.final_results = []
            if hasattr(st.session_state, 'selected_image'):
                delattr(st.session_state, 'selected_image')
            st.success("모든 데이터가 삭제되었습니다.")
            st.experimental_rerun()

    with col3:
        if st.button("🔄 이미지 목록 초기화"):
            st.session_state.uploaded_images = []
            st.session_state.analysis_results = {}
            if hasattr(st.session_state, 'selected_image'):
                delattr(st.session_state, 'selected_image')
            st.success("이미지 목록이 초기화되었습니다.")
            st.experimental_rerun()

    with col4:
        if st.button("📥 엑셀로 다운로드") and st.session_state.final_results:
            df_export = pd.DataFrame(st.session_state.final_results)

            # 컬럼명을 한글로 변경
            df_export_kr = df_export.rename(columns=COLUMN_NAMES_KR)

            # 엑셀 파일 생성
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export_kr.to_excel(writer, index=False, sheet_name='선정성_탐지_결과')

            excel_data = output.getvalue()

            # 다운로드 버튼
            st.download_button(
                label="📁 Excel 파일 다운로드",
                data=excel_data,
                file_name=f"선정성_탐지_결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()