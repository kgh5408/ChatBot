import os 
import logging
import pathlib
import tempfile
from langchain.document_loaders import(
    PyPDFLoader,
    # TextLoader,
    # UnstructuredEPubLoader,
    UnstructuredWordDocumentLoader
)
from typing import List
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import (
    ConversationalRetrievalChain,
    FlareChain,
    OpenAIModerationChain,
    SimpleSequentialChain,
)
from langchain.chains.base import Chain
from utils import *
from langchain_openai import ChatOpenAI

# ----- 문서 로드 부분
# 문서 로드 예외 처리 클래스
class DocumentLoaderException(Exception):
    pass

# 다양한 확장자를 지원하는 문서 로더 클래스 정의
class DocumentLoader(object):
    supported_extenstions = {
        ".pdf": PyPDFLoader,
        # ".txt": TextLoader,
        # ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        # ".doc": UnstructuredWordDocumentLoader
    }

# 파일을 로드하여 문서 리스트로 반환하는 함수
def load_document(temp_filepath: str) -> List[Document]:
    ext = pathlib.Path(temp_filepath).suffix  # 파일 확장자 추출
    loader = DocumentLoader.supported_extenstions.get(ext)  # 확장자에 맞는 로더 선택
    if not loader:
        raise DocumentLoaderException(
            f"지원되지 않는 파일 확장자: {ext}, 이 파일 형식은 로드할 수 없습니다."
        ) 
    
    # 파일 로드 및 문서 반환
    loaded = loader(temp_filepath)
    docs = loaded.load()  # 문서 데이터를 가져옴
    logging.info(docs)
    return docs

# ----- 등기부등본 가져오기
def extract_document1(
        uploaded_files,
        type="full"
):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()  # 임시 파일 생성
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    # 문서 텍스트 분할
    full_content = " ".join([doc.page_content for doc in docs])
    if type == "full":
        return full_content
    elif type == "chunk":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)  # chunk_overlap: 청크로 잘랐을때 겹치는 부분 글자 수
        splits = text_splitter.split_documents(docs)
        chunk = " ".join([split.page_content for split in splits])
        return chunk
    elif type == "summarize":
        pass


# ----- Langchain
# 문서 검색기 설정 함수
def configure_retriever(
        docs: List[Document],
        use_compression: bool = False  # 정보를 압축하는 기술
) -> BaseRetriever:
    # 문서 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)  # chunk_overlap: 청크로 잘랐을때 겹치는 부분 글자 수
    splits = text_splitter.split_documents(docs)

    # 임베딩 생성 및 백터 DB 저장
    embeddings = OpenAIEmbeddings()
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        }
    )
    if not use_compression:
        return retriever
    
    # 임베딩 필터 설정
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.4
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever,
    )

# 검색 체인 설정 함수
def configure_chain(llm, prompt, retriever: BaseRetriever, use_flare: bool = True) -> Chain:
    params = dict(
        llm=llm,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        # max_token_limit=400,
        # prompt=prompt,
        combine_docs_chain_kwargs={"prompt": prompt}
        # condense_question_prompt=prompt,
    )

    if use_flare:
        # FlareChain 사용
        return FlareChain.from_llm(
            **params
        )
    return ConversationalRetrievalChain.from_llm(
        **params
    )

# 파일 업로드 후 체인 설정 함수
def configure_retrieval_chain(
        llm,
        prompt,
        use_compression: bool = False,
        use_flare: bool = False,
) -> Chain:
    docs = []
    for file in os.listdir("./data"):
        filepath = os.path.join("./data", file)
        try:
            docs.extend(load_document(filepath))
        except Exception as e:
            print(f"Error loading file {file}: {e}")

    retriever = configure_retriever(docs=docs, use_compression=use_compression)
    chain = configure_chain(llm, prompt,  retriever=retriever, use_flare=use_flare)

    return chain
   
