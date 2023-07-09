import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

HUGGINGFACE_API_KEY = st.secrets["huggingface_api_key"]

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="text-generation",
    model_kwargs={
        "min_length": 200,
        "max_length": 2000,
        "temperature": 0.2,
        "max_new_tokens": 200,
        "num_return_sequences": 1
    }
)

template = """
    Answer the question as truthfully as possible using the provided text. 
    If the answer is not contained within the text below, say "I don't know"
    Context:{context}
    >>QUESTION<<{question}
    >>ANSWER<<
""".strip()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)


@st.cache_resource
def load_chroma_db(persist_directory, _embedding):
    
    if os.path.isfile('./db/index/*.pkl'):
        vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=_embedding
        )
    else:
        loader = DirectoryLoader(
            './transcripts/',
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=_embedding,
            persist_directory=persist_directory
        )
        
    return vectordb


def generate_response_db(query_text):
    if query_text is not None:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vectordb.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        result = qa({"query": query_text})
        
        return result


st.title('Masters of Advertising: Exam Bot :sunglasses:')

st.subheader("Question")
query_text = st.text_input(
    'Text input title:',
    label_visibility="collapsed",
    placeholder='e.g. What is Future Demand?'
)

# Load embeddings
persist_directory = 'db'
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = load_chroma_db(persist_directory, embedding)

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
    
    with st.expander("**See sources**"):
        for document in response["source_documents"]:
            st.text(document.metadata["source"])
            st.info(document.page_content)
