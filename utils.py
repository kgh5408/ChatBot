import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np
import re

# ---- User message 오른쪽 출력
def markdown():
    st.markdown(
    """
    <style>
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

# ---- 이전 대화 기록을 출력
def print_messages():
    if "messages" in st.session_state and len(st.session_state['messages']) > 0:
        for chat_message in st.session_state['messages']:
            st.chat_message(chat_message.role).write(chat_message.content)

# ---- 토큰 하나당 하나씩 출력
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# ---- 세션 ID를 기반으로 새션 기록을 가져오는 함수 
# 1. Local
# def get_session_history_local(session_ids:str) -> BaseChatMessageHistory:
#     if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
#         # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
#         st.session_state["store"][session_ids] = ChatMessageHistory()
#     return st.session_state["store"][session_ids]

# 2. Persistent store (Redis)
# 라이브러리 설치: pip install -qU redis
# Docker run: docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
# REDIS_URL = "redis://localhost:6379/0"
# def get_session_history_redis(session_ids:str) -> RedisChatMessageHistory:
#     return RedisChatMessageHistory(session_ids,url=REDIS_URL)

# 3. Local2
def init_memory():
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        # output_key='answer'
    )

# 대화기록 메모리 저장
MEMORY = init_memory()


# ----- 실거래가
def extreact_nearby_price(address_doro, area):
    df = pd.read_pickle("./실거래가.pkl")
    address_doro = address_doro.split(",")[0]
    split_address = address_doro.split(" ")

    sigu = " ".join(split_address[:2])
    logil_idx = [i for i, word in enumerate(split_address) if "로" in word and "길" in word]
    if len(logil_idx) == 0:
        logil_idx = [i for i, word in enumerate(split_address) if word.endswith("로") or word.endswith("길")]
        logil = "".join([split_address[logil_idx[0]],split_address[logil_idx[1]]])
        doro = " ".join([logil, split_address[int(logil_idx[1]+1)]])
    else:
        doro = " ".join([split_address[logil_idx[0]], split_address[logil_idx[0]+1]])
    doro_split = doro.split(" ")
    print(doro)

    # 시군구 일치    
    df = df.loc[df["시군구"].str.contains(sigu, na=False),:]

    거래금액 = df.loc[df["도로명"]==doro,"거래금액"]
    df2 = df.loc[df["도로명"].str.contains(doro_split[0], na=False),:].copy()
    df3 = df.loc[df["도로명"].str.contains(re.search(r'.*?로', doro_split[0]).group(), na=False),:].copy()
    # 도로명 전체 일치
    if len(거래금액) != 0:
        price = np.mean(거래금액)
    # 도로명 "~길"까지 일치
    elif len(df2) != 0:
        temp = " ".join([doro_split[0], doro_split[1].split("-")[0]])
        df_temp = df.loc[df["도로명"].str.contains(temp, na=False),:].copy()
        # "~길 숫자"까지 일치
        if len(df_temp) != 0:
            idx = abs(df_temp["전용면적"] - area).idxmin()
            price = float(df_temp.loc[idx, "거래금액"])
        else:
            idx = abs(df2["전용면적"] - area).idxmin()
            price = float(df2.loc[idx, "거래금액"])
    # 도로명 "~로"까지 일치
    elif len(df3) != 0:
        idx = abs(df3["전용면적"] - area).idxmin()
        price = float(df3.loc[idx, "거래금액"])
    else:
        price = "알수없음"
    return int(price * 10000)
