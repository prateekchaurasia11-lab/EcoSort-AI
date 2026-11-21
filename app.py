"""Run: streamlit run app.py"""
import hashlib
import io
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
from Src.demo_service import GUIDANCE, predict_upload, validate_upload

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='EcoSort AI', page_icon='♻️', layout='wide')
st.markdown('''<style>.stApp{background:#f5f8f4}h1,h2,h3{color:#174c34}.block-container{max-width:1120px;padding-top:2.5rem}div[data-testid="stMetric"]{background:white;padding:1rem;border-radius:14px}</style>''', unsafe_allow_html=True)
st.caption('ECOSORT AI  /  WASTE CLASSIFICATION')
st.title('A clearer next step for your waste.')
st.write('Photograph one waste item, explore its predicted material, and check how to sort it locally.')

@st.cache_resource
def load_predictor(checkpoint, protocol, checkpoint_stamp, protocol_stamp):
    from Src.predict_keras import E2Predictor
    return E2Predictor(checkpoint, protocol)

checkpoint = Path(os.environ.get('ECOSORT_CHECKPOINT', str(ROOT / 'Models/e2/best.keras')))
protocol = Path(os.environ.get('ECOSORT_PROTOCOL', str(ROOT / 'Models/e2/protocol.json')))
with st.sidebar:
    st.header('About this demo')
    st.write('E2 · MobileNetV2 · Six material classes')
    st.write('Best for one clearly visible waste item. Mixed materials, backgrounds and unfamiliar objects can be misclassified.')
    st.caption('The model always selects one of six classes. Unknown-object rejection and score calibration are pending.')
    with st.expander('Verified baseline results'):
        st.write('TrashNet test: 317/382 correct (82.98%). Macro F1: 0.8013. TensorFlow decoder. This is a historical benchmark, not a guarantee for your photos.')

left, right = st.columns([1.05, 1], gap='large')
with left:
    st.subheader('1. Add an image')
    upload = st.file_uploader('JPEG or PNG · up to 10 MB · below 20 megapixels', type=['jpg', 'jpeg', 'png'])
    raw = upload.getvalue() if upload else None
    valid = False
    if raw:
        try:
            validate_upload(raw)
            st.image(Image.open(io.BytesIO(raw)), width='stretch')
            valid = True
        except ValueError as exc:
            st.error(str(exc))
    digest = hashlib.sha256(raw).hexdigest() if raw else None
    if st.session_state.get('image_digest') != digest:
        st.session_state.pop('prediction', None)
        st.session_state['image_digest'] = digest
    if not checkpoint.is_file() or not protocol.is_file():
        st.info('Model assets are not configured yet. Follow DEMO_QUICKSTART.md to install the verified E2 bundle.')
    if st.button('Classify image', type='primary', disabled=not valid or not checkpoint.is_file() or not protocol.is_file()):
        st.session_state.pop('prediction', None)
        try:
            with st.spinner('Inspecting the material…'):
                predictor = load_predictor(str(checkpoint), str(protocol), checkpoint.stat().st_mtime_ns, protocol.stat().st_mtime_ns)
                st.session_state['prediction'] = predict_upload(predictor, raw)
        except Exception:
            st.error('Classification failed. Check the model bundle and image, then try again. No result was saved.')
with right:
    st.subheader('2. Explore the result')
    result = st.session_state.get('prediction')
    if result:
        st.metric('Predicted material', result['class'].title())
        st.metric('Model score · uncalibrated', f"{result['model_score']:.1%}")
        st.caption('This score is not a measured probability that the item is recyclable. Confirm the material yourself.')
        st.subheader('Suggested next step')
        st.write(GUIDANCE[result['class']])
        st.caption('General sorting guidance. Local collection rules determine what is accepted.')
        scores = pd.DataFrame({'Material': list(result['scores']), 'Model score': list(result['scores'].values())}).set_index('Material')
        st.bar_chart(scores, horizontal=True)
        st.download_button('Download prediction JSON', json.dumps(result, indent=2), 'ecosort_prediction.json', 'application/json')
        with st.expander('Model and preprocessing details'):
            st.json({key: result[key] for key in ['checkpoint_sha256', 'decoder', 'preprocessing_version', 'calibrated']})
    else:
        st.info('Your prediction and sorting guidance will appear here.')
st.divider()
st.caption('Research prototype · E2 baseline · E1 recovery, expanded-data experiments, calibration and detector integration remain pending.')
