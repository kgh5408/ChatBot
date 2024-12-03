"""
[git-streamlit 연동]
- https://yeomss.tistory.com/301

[streamlit 강의]
- https://www.youtube.com/watch?v=ZVmLe3odQvc
- https://www.youtube.com/watch?v=VtS8yF2ItgI

"""
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
import yaml
import streamlit_authenticator as stauth
from utils import *
from documents import *
from langchain.prompts.chat import (
HumanMessagePromptTemplate,
SystemMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from streamlit.external.langchain import StreamlitCallbackHandler
import re
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('api_key')
os.environ["OPENAI_API_KEY"] = api_key


st.set_page_config(page_title="전세 사기 방지 ChatBot", page_icon="🏠")
st.title("🏠 전세 사기 방지 ChatBot")

# login (https://github.com/mkhorasani/Streamlit-Authenticator)
with open('config.yaml', encoding='utf-8') as file:
    config = yaml.load(file, Loader=stauth.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    # config['pre-authorized']
)

st.session_state['authentication_status'] = True  # !!!!!!!!!!! 나중에 지우기

if st.session_state['authentication_status'] is None:
    try:
        authenticator.login('main')
        if st.session_state['authentication_status']:
            st.rerun()  
    except Exception as e:
        st.error(e)
    # st.warning('Please enter your username and password')

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')

elif st.session_state['authentication_status']:
    markdown()

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    if "store" not in st.session_state:
        st.session_state["store"] = dict()

    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None

    with st.sidebar:
        session_id = st.session_state["name"]

        st.write(f'사용자 - {st.session_state["name"]}')
        authenticator.logout()
        
        # 등기부등본 업로드
        file_uploader_key = f"file_uploader_{st.session_state.get('reset_counter', 0)}"
        uploaded_files = st.file_uploader(
            label="등기부등본",
            type=list(DocumentLoader.supported_extenstions.keys()),
            accept_multiple_files=True,
            key=file_uploader_key,
        )
        st.session_state["uploaded_file"] = uploaded_files

        if not st.session_state["uploaded_file"]:
            st.info("등기부등본을 업로드 하세요.")
            st.stop()

        # 등기부등본 내용 추출
        st.session_state["document1"] = extract_document1(st.session_state["uploaded_file"], type="full")

        # 대화기록 초기화
        clear_btn = st.button("대화기록 초기화")
        if clear_btn:
            st.session_state['messages'] = []
            st.session_state["store"] = dict()
            st.session_state["uploaded_file"] = None
            if "reset_counter" not in st.session_state:
                st.session_state["reset_counter"] = 0
            st.session_state["reset_counter"] += 1
            st.rerun()

    # 이전 대화기록 출력 
    print_messages()

    # 처음 대화 내용 고정 - 대화 내용 고정
    if (len(st.session_state['messages'])==0) & (st.session_state["uploaded_file"] is not None):
        # 전용 면적 추출
        area = 0
        for text in st.session_state["document1"].split("건 물 내 역"):
            if "㎡" in text:
                text2 = text.split("대지권")[0]
                numbers = re.findall(r'\s*(\d+\.?\d*\s?)㎡', text2)
                for number in numbers:
                    area += float(number)
                break
        area = round(area, 4)

        user_input = f"""
            등기부등본:{st.session_state["document1"]} 
            
            위 등기부등본 내용을 구체적으로 말해주세요.
            
            [답변형식]
            본 등기부등본의 내용을 정리하여 말씀 드리겠습니다.
            - 지번 주소:
            - 두로명 주소:
            - 전용 면적: {area}㎡
            - 총 근저당: 
            - 신탁 여부:
            - 압류 현황:
            - 등등
        """  # !!!!! 답변형식을 지정해주면 좋을듯!
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
            prompt_template = """
                {chat_history}
                질문: {input}
                답변:
            """
            conv_llm = ConversationChain(llm=llm, 
                                         memory=MEMORY,
                                         prompt=PromptTemplate(template=prompt_template, input_variables=["chat_history", "input"]))
            response = conv_llm.predict(input=user_input)
            st.session_state['messages'].append(
                ChatMessage(role='assistant', content=response)
            )

            # 주변 시세 추출
            doro_match = re.search(r"도로명 주소:\s*(.+)", response)
            address_doro = doro_match.group(1) if doro_match else None

            st.session_state["price"] = extreact_nearby_price(address_doro, area)

    user_input = st.chat_input("메세지를 입력해 주세요.")
    if user_input:
        # 사용자 입력
        st.chat_message("user").write(f"{user_input}")
        st.session_state['messages'].append(ChatMessage(role='user', content=user_input))
        
        # AI 딥변
        assistant = st.chat_message("assistant")
        stream_handler = StreamlitCallbackHandler(assistant)
        # container = st.empty()
        with st.chat_message("assistant"):
            # 1. LLM 생성
            llm = ChatOpenAI(streaming=True)
            
            # 2. 프롬프트 생성
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        # "등기부등본 내용은 다음과 같습니다: {document}"
                        # "전세사기와 관련된 질문에 친절히 답변하세요."
                        "너는 부동산 전문가야"
                    ),
                    # # 대화 기록을 변수로 사용, history가 MessageHistory의 key가 됨.
                    # MessagesPlaceholder(variable_name="history"),
                    ("human","{question}"),
                ]
            )
    
            system_message = SystemMessagePromptTemplate.from_template(
                f"본 집의 주변 시세: {st.session_state['price']}원"+
                """
                참고 내용: {context}
                이전 대화 내용: {chat_history} 
                
                너는 위 내용을 기반하여 답변하면 된다.
                """
            )
            human_message = HumanMessagePromptTemplate.from_template(
                """
                {question}
                """
            )
            prompt = ChatPromptTemplate.from_messages([system_message, human_message])

            
            # 3. Chain 생성
            # # 콘텐츠 검열 기능 
            # moderation = OpenAIModerationChain()
            # chain = prompt | llm | moderation

            conv_chain = configure_retrieval_chain(llm, prompt)

            response = conv_chain.run({"question": user_input, "chat_history": MEMORY.chat_memory.messages},
                                      callbacks=[stream_handler]
                                      )

            msg = response

            # chain_with_memory = (
            #     RunnableWithMessageHistory(
            #         chain,
            #         get_session_history_local,
            #         input_messages_key="input",  # 사용자 질문 키
            #         history_messages_key="history",  # 기록 메세지 키
            #     )
            # )
            
            # response = chain_with_memory.invoke(
            #     {"document": document1, "input": user_input},
            #     # {"input": user_input},
            #     config={"configurable": {"session_id": session_id}}
            # )

            # # msg = response.content
            # msg = response['output'].content  # OpenAIModerationChain 사용 시

            # container.markdown(msg)
            st.write(msg)
            st.session_state['messages'].append(
                ChatMessage(role='assistant', content=msg)
            )






