import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
import os
from PIL import Image
import io
import base64

# 페이지 설정
st.set_page_config(
    page_title="선정성 탐지 임계값 최적화 도구",
    page_icon="🔍",
    layout="wide"
)

# 세션 상태 초기화
if 'results' not in st.session_state:
    st.session_state.results = []

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = {}

def generate_mock_yolo_results():
    """YOLO 모델 분석 결과 모의 데이터 생성"""
    classes = ['EXPOSED_BREAST_F', 'EXPOSED_BREAST_M', 'EXPOSED_BUTTOCKS',
               'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M']

    detected_count = random.randint(0, 3)
    detected_classes = random.sample(classes, min(detected_count, len(classes))) if detected_count > 0 else []

    confidences = {}
    for cls in classes:
        if cls in detected_classes:
            confidences[cls.lower() + '_confidence'] = round(random.uniform(0.3, 0.95), 3)
        else:
            confidences[cls.lower() + '_confidence'] = 0.0

    highest_confidence = max(confidences.values()) if confidences else 0.0
    highest_class = max(confidences.items(), key=lambda x: x[1])[0].replace('_confidence', '') if highest_confidence > 0 else ""

    return {
        'detected_objects_count': detected_count,
        'detected_classes': ', '.join(detected_classes),
        'highest_confidence_class': highest_class,
        'highest_confidence_score': highest_confidence,
        **confidences
    }

def generate_mock_skin_analysis():
    """살색 분석 결과 모의 데이터 생성"""
    total_skin_ratio = round(random.uniform(5.0, 60.0), 2)

    return {
        'total_skin_ratio': total_skin_ratio,
        'cr_channel_concentration': round(random.uniform(20.0, 40.0), 2),
        'cb_channel_concentration': round(random.uniform(40.0, 60.0), 2),
        'cr_skin_pixel_ratio': round(random.uniform(0.2, 0.4), 3),
        'cb_skin_pixel_ratio': round(random.uniform(0.4, 0.6), 3),
        'skin_threshold_25_result': total_skin_ratio >= 25,
        'skin_threshold_40_result': total_skin_ratio >= 40
    }

def analyze_image(model_name, image_file, confidence_threshold):
    """이미지 분석 수행 (모의)"""
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
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
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
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("1. 모델 및 이미지 설정")

        # 모델 선택
        model_options = [
            "best_640n_0522.onnx",
            "best_320n_0415.onnx",
            "experimental_model_v2.onnx"
        ]
        selected_model = st.selectbox("분석할 모델 선택", model_options)

        # 신뢰도 임계값
        confidence_threshold = st.slider("신뢰도 임계값", 0.1, 0.9, 0.5, 0.1)

        # 이미지 업로드
        uploaded_files = st.file_uploader(
            "분석할 이미지 선택 (.webp, .jpg, .png)",
            type=['webp', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)}개 이미지가 선택되었습니다.")

            # 이미지 미리보기
            if st.checkbox("이미지 미리보기"):
                cols = st.columns(min(len(uploaded_files), 4))
                for i, file in enumerate(uploaded_files[:4]):
                    with cols[i]:
                        image = Image.open(file)
                        st.image(image, caption=file.name, use_column_width=True)
                if len(uploaded_files) > 4:
                    st.info(f"처음 4개 이미지만 미리보기됩니다. (총 {len(uploaded_files)}개)")

    with col2:
        st.header("2. 분류 설정")

        # 정탐/오탐 선택
        classification = st.radio(
            "이미지 분류",
            ["정탐", "오탐"],
            help="선택한 이미지들이 정탐(올바른 탐지)인지 오탐(잘못된 탐지)인지 선택하세요."
        )

        # 오탐 특징 입력 (오탐인 경우에만)
        false_positive_features = ""
        if classification == "오탐":
            false_positive_features = st.text_area(
                "오탐 특징 설명",
                placeholder="예: 음식 이미지, 도넛 형태, 살색과 유사한 색상 등",
                help="오탐이 발생한 이유나 특징을 입력하세요."
            )

        # 분석자 메모
        analyst_notes = st.text_area(
            "분석자 메모 (선택사항)",
            placeholder="추가적인 관찰 사항이나 메모"
        )

    st.markdown("---")

    # 분석 실행
    if st.button("🚀 분석 실행", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("분석할 이미지를 선택해주세요.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_container = st.container()

            for i, file in enumerate(uploaded_files):
                status_text.text(f"분석 중... ({i+1}/{len(uploaded_files)}) {file.name}")
                progress_bar.progress((i + 1) / len(uploaded_files))

                # 이미지 분석 수행
                analysis_result = analyze_image(selected_model, file, confidence_threshold)

                # 분류 정보 추가
                analysis_result.update({
                    'id': len(st.session_state.results) + 1,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'classification': classification,
                    'false_positive_features': false_positive_features,
                    'analyst_notes': analyst_notes
                })

                # 결과 저장
                st.session_state.results.append(analysis_result)

            status_text.success(f"✅ {len(uploaded_files)}개 이미지 분석 완료!")
            progress_bar.empty()

            # 결과 표시
            with results_container:
                st.header("📋 분석 결과")

                # 최근 분석 결과만 표시
                recent_results = st.session_state.results[-len(uploaded_files):]
                df_recent = pd.DataFrame(recent_results)

                # 주요 결과 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_confidence = df_recent['highest_confidence_score'].mean()
                    st.metric("평균 신뢰도", f"{avg_confidence:.3f}")

                with col2:
                    avg_skin_ratio = df_recent['total_skin_ratio'].mean()
                    st.metric("평균 살색 비율", f"{avg_skin_ratio:.1f}%")

                with col3:
                    detected_count = df_recent['detected_objects_count'].sum()
                    st.metric("총 탐지 객체", detected_count)

                with col4:
                    threshold_pass = df_recent['skin_threshold_25_result'].sum()
                    st.metric("임계값(25%) 통과", f"{threshold_pass}/{len(df_recent)}")

                # 상세 결과 테이블
                st.subheader("상세 분석 결과")
                display_columns = [
                    'file_name', 'classification', 'highest_confidence_score',
                    'total_skin_ratio', 'detected_objects_count', 'detected_classes'
                ]
                st.dataframe(df_recent[display_columns], use_container_width=True)

    # 저장된 결과 관리
    st.markdown("---")
    st.header("💾 데이터 관리")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 전체 결과 보기"):
            if st.session_state.results:
                df_all = pd.DataFrame(st.session_state.results)
                st.dataframe(df_all, use_container_width=True)
            else:
                st.info("저장된 결과가 없습니다.")

    with col2:
        if st.button("🗑️ 모든 데이터 삭제"):
            st.session_state.results = []
            st.success("모든 데이터가 삭제되었습니다.")
            st.experimental_rerun()

    with col3:
        if st.button("📥 엑셀로 다운로드") and st.session_state.results:
            df_export = pd.DataFrame(st.session_state.results)

            # 엑셀 파일 생성
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='선정성_탐지_결과')

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