import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

# 왼쪽 사이드 바 
with st.sidebar:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    '[본인의 OpenAI API KEY를 입력](https://platform.openai.com/api-keys)'

# 오른쪽 메인 화면
st.title('🤖 Chatbot')

# 초기 질문 설정(한번도 아직 대화를 안했다면)
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role':'assistant','content':'How can I help you?'}]


# 대화 기록을 출력
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# 사용자의 입력을 받아 대화 기록에 추가하고 AI가 응답을 생성 
if prompt := st.chat_input():
    if not openai_api_key:
        st.info('Please add you OpenAI API key to continue.')
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({'role': 'user','content': prompt})

    st.chat_message('user').write(prompt) # 화면에 보여준다
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=st.session_state.messages
    )
    msg = response.choices[0].message.content
    st.session_state.messages.append({'role':'assistant','content': msg})

    st.chat_message('assistant').write(msg)