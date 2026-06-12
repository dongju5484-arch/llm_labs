import streamlit as st
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, Document
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv
import os

# 환경 변수 설정 (api key 설정)
load_dotenv()
# OpenAI API key 설정
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# 현재 파일의 디렉토리를 기준으로 상대 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '.', 'data')

# Streamlit 앱 설정
def setup_streamlit_page():
    st.set_page_config(page_title='LlamaIndex QA', page_icon='🦙')
    st.title('RAG 관련 Q&A')

# LLM 설정
def initialize_llm_and_settings():
    llm = OpenAI(
        temperature=0.5,
        model='gpt-4o',
        max_tokens=200,
        context_window=4096,
    )
    Settings.llm = llm
    return llm

# 업로드된 파일 처리
def process_uploaded_files(uploaded_files):
    """업로드된 파일들을 처리하여 문서 리스트를 생성하는 함수"""
    documents = []
    pdf_reader = PDFReader()

    for file in uploaded_files:
        try:
            content = file.read()  # 파일 내용을 바이트로 읽기

            # 텍스트 파일의 경우
            if file.type == 'text/plain':
                text_content = content.decode('utf-8') # 유니코드 디코드 변환
                doc = Document(text=text_content, metadata={'filename':file.name})
                documents.append(doc)
            # PDF 파일인 경우
            elif file.type == 'application/pdf':
                # 임시 파일로 저장
                with open(f'temp_{file.name}', 'wb') as f:
                    f.write(content)
                # PDF 파일 읽기
                pdf_docs = pdf_reader.load_data(f'temp_{file.name}')
                documents.extend(pdf_docs)
                # 임시 파일 삭제
                os.remove(f'temp_{file.name}')
            else:  # 그 외 파일이라면
                st.error(f'지원하지 않는 파일 형식입니다: {file.name}')
        
        except Exception as e:
            st.error(f'파일 처리 중 오류 발생: {file.name} - {str(e)}')

    return documents

# 채팅 엔진
def initialize_chat_engine(index):
    chat_engine = index.as_chat_engine(
        chat_mode='condense_plus_context',
        verbose=True,
        system_prompt='''
        당신은 업로드된 문서를 기반으로 답변하는 AI 어시스턴트입니다.
        이전 대화 내용을 고려하여 답변하되, 필요한 경우에만 문서를 검색하여 답변해주세요.
        업로드된 문서에서 찾을 수 없는 내용은 솔직히 모른다고 말씀해주세요.
        '''
    )
    return chat_engine


if __name__ == '__main__':
    setup_streamlit_page()

    # 파일 업로드 위젯
    st.sidebar.header('문서 업로드')
    uploaded_files = st.sidebar.file_uploader(
        '텍스트 파일이나 PDF 파일을 업로드하세요',
        accept_multiple_files=True,  # 파일 여러개 선택 가능
        type=['txt', 'pdf']
    )

    # 업로드된 파일 처리 및 인덱스 생성
    if uploaded_files:
        with st.spinner('문서를 처리하고 있습니다...'):
            initialize_llm_and_settings()  # 함수 호출 - llm 불러오는 함수
            documents = process_uploaded_files(uploaded_files)  # 업로드된 파일처리하는 함수

            if documents: # 문서가 있다면
                # 문서 정보 표시
                st.sidebar.success(f'처리된 문서 수 : {len(documents)}')

                # 인덱스 생성
                index = VectorStoreIndex.from_documents(documents)
                st.session_state.chat_engine = initialize_chat_engine(index)
                st.success('문서 처리가 완료되었습니다!')

            else:
                st.error('처리할 수 있는 문서가 없습니다!')


    # 채팅 기록 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content']) # 메세지 내용이 마크다운 형식으로 보여진다

    # 채팅 엔진이 준비된 경우에만 입력 활성화
    if 'chat_engine' in st.session_state:
        # 사용자 입력
        if prompt := st.chat_input('무엇이 궁금한가요??'):
            st.session_state.messages.append({'role':'user', 'content':prompt})
            with st.chat_message('user'):
                st.markdown(prompt) # 사용자의 프롬프트가 마크다운 형식으로 보여진다

            # AI 응답 생성
            with st.chat_message('assistant'):
                with st.spinner('Thinking...'):
                    response = st.session_state.chat_engine.chat(prompt)
                    st.markdown(response.response) # 답변이 마크다운 형식으로 보여진다
                    st.session_state.messages.append(
                        {'role':'assistant', 'content':response.response}
                    )  # 답변이 세션에 추가된다

            # 디버그 정보
            with st.expander('Debug Info'):
                st.write(f'Response Type: {type(response)}')
                st.write('Source Nodes:', response.source_nodes if hasattr(response, 'source_nodes') else None)
    
    else:
        st.info('문서를 업로드하면 질문할 수 있습니다.')

    # 채팅 초기화 버튼
    if st.sidebar.button('채팅 초기화'):
        st.session_state.messages = []
        st.rerun()


    
    