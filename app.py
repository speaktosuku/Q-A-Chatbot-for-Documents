import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 👉 Load environment variables
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 👉 Load PDF Document
pdf_loader = PyPDFLoader("PDF Sample.pdf")
pdf_docs = pdf_loader.load()

# 👉 Load Word Document (if needed)
docx_loader = UnstructuredWordDocumentLoader("Docs Sample.docx")
docx_docs = docx_loader.load()

# 👉 Combine all documents
all_docs = pdf_docs + docx_docs

# 👉 Split Text into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
split_docs = splitter.split_documents(all_docs)

# 👉 Generate Embeddings & Store in FAISS
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_docs, embeddings)

# 👉 Setup LangChain Retriever & Memory
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 👉 RAG Chain (Conversational Retrieval)
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# 👉 Streamlit UI
st.title("📚 Q&A Chatbot for Document Retrieval")

# 👉 User Query Input
user_query = st.text_input("🔍 Ask a question based on the documents:")

if user_query:
    response = rag_chain.run(user_query)
    st.write("### 🤖 Answer:")
    st.write(response)








