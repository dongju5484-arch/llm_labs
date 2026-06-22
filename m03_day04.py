# =====================================================
# 1. 라이브러리 불러오기
# =====================================================
import os
import json
import base64
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# =====================================================
# 2. 환경 설정 - api키 불러오기, 상수개념(이름), 웹페이지 설정
# =====================================================
load_dotenv()

APP_TITLE = 'Invoice / receipt Analyzer'
MODEL = 'gpt-4o-mini'
OUTPUT_DIR = Path('outputs') # outputs라는 폴더 경로 지정
OUTPUT_DIR.mkdir(exist_ok=True) # 폴더 만든다 - 이미 있으면 그냥 넘어간다

st.set_page_config(page_title=APP_TITLE, page_icon='🧾', layout='wide')

# =====================================================
# 3. 함수 정의 - openai api key
# =====================================================
def get_client() -> OpenAI:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('api 키가 없습니다! .env 파일을 확인하세요.')
    return OpenAI(api_key=api_key)

# =====================================================
# 4. 함수 정의 - 이미지를 불러와서 컴퓨터가 읽을 수있도록 인코딩
# =====================================================
def image_to_data_url(uploaded_file) -> str:
    raw = uploaded_file.getvalue() # 원시 데이터 저장
    b64 = base64.b64encode(raw).decode('utf-8')
    mime = uploaded_file.type or 'image/png'
    return f'data:{mime};base64, {b64}'