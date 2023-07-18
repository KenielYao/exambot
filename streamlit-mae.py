import os
from typing import List
import string
import requests

import streamlit as st

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain.retrievers import BM25Retriever, MergerRetriever

HUGGINGFACE_API_KEY = st.secrets["huggingface_api_key"]


llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="text-generation",
    model_kwargs={
        "temperature": 0.3, # 0.2 - 0.4
        "repetition_penalty": 1.2, # 1.2 recommended, >1 discourages repetition
        "encoder_repetition_penalty": 1.2,
        # new_tokens = tokens used, excluding the prompt and context
        "min_new_tokens": 50,
        "max_new_tokens": 250, 
        "length_penalty": 0, # > 0 = longer, < 0 = shorter
        "num_return_sequences": 1
    }
)

template = """
    Context:{context}
    
    You are a marketing professor responding to student questions on the 
    evidence of advertising effectiveness. Answer the question as truthfully as 
    possible using the context provided above. 
    If the answer is not contained within the text below, say "I don't know."
    
    >>QUESTION<<{question}
    >>ANSWER<<
""".strip()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

@st.cache_resource()
def get_stopwords_array(url):
    response = requests.get(url)
    response.raise_for_status()

    stopwords_text = response.text
    stopwords_array = stopwords_text.split('\n')

    return stopwords_array

stopwords_url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt'
stopwords_array = get_stopwords_array(stopwords_url)

def custom_preprocessing_func(text: str) -> List[str]:
    
    words = text.lower().split()
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords_array]
    
    # Remove punctuation
    words = [word for word in words if word not in string.punctuation]
    
    # Remove remaining non-alphabetic tokens
    words = [word for word in words if word.isalpha()]

    return words

@st.cache_resource()
def generate_documents_from_directory():
    loader = DirectoryLoader(
        './transcripts/',
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, # 1000
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(documents)
    
    return documents

@st.cache_resource
def load_chroma_db(persist_directory, _embedding, _documents):
    
    if os.path.exists('./db/index/'):
        vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=_embedding
        )
    else:
        vectordb = Chroma.from_documents(
            documents=_documents,
            embedding=_embedding,
            persist_directory=persist_directory
        )
        
    return vectordb

@st.cache_resource
def load_bm25(persist_directory, _documents, should_preprocess):
        
    if should_preprocess:
        bm25 = BM25Retriever.from_documents(_documents, preprocess_func = custom_preprocessing_func)
    else:
        bm25 = BM25Retriever.from_documents(_documents)

    return bm25


def generate_response_db(query_text):
    
    lotr = MergerRetriever(retrievers=[
        chroma_mini.as_retriever(),
        retriever_bm25
    ]) 
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=lotr,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    try:
        result = qa({"query": query_text})
    except ValueError:
        result = {result: "It's taking longer than expected to load. Please try again in a few seconds..."}
        
    return result


st.title('Master of Advertising Effectiveness: Exam Bot :sunglasses:')
st.divider()

st.subheader("Question")
query_text = st.text_input(
    'Text input title:',
    label_visibility="collapsed",
    placeholder='e.g. What is Future Demand?'
)
st.caption("To get the best response, make your question as specific as possible.")

# Load embeddings
persist_directory = 'db'
documents = generate_documents_from_directory()

embed_mini = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_mini = load_chroma_db(persist_directory, embed_mini, documents)

should_preprocess = True
retriever_bm25 = load_bm25(persist_directory, documents, should_preprocess)

# Form input and query
result = []
source_documents = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not query_text)
    
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response_db(query_text)
            result.append(response["result"])

if len(result):    
    st.subheader("Answer")
    st.info(response["result"])
    
    with st.expander("Answers aren't always reliable. Click below to see which lecture notes Exambot referenced."):
        for document in response["source_documents"]:
            st.text(document.metadata["source"])
            st.info(document.page_content)
