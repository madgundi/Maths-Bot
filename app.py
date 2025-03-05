import streamlit as st
import os
import tempfile
import logging
import pytesseract
from PIL import Image
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ✅ Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows only

# ✅ Secure API Key Handling
API_KEY = "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"
os.environ["GROQ_API_KEY"] = API_KEY

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Define Math Chatbot System Prompt
SYSTEM_PROMPT = """
You are a highly skilled mathematician specializing in advanced concepts such as differential geometry, topology, and abstract algebra.
You provide detailed, structured, and rigorous explanations, using LaTeX formatting where necessary.
"""

# ✅ Initialize Chat Model
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=API_KEY)

# ✅ Class for Document and Image Processing
class MultiFormatRAG:
    def __init__(self):
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def load_documents(self, directory_path):
        documents = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in self.loader_map:
                try:
                    loader = self.loader_map[ext](file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
        return documents

    def analyze_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")  # Ensure correct format
            text = pytesseract.image_to_string(image)
            if not text.strip():
                return "No text detected in the image."
            return text.strip()
        except Exception as e:
            logger.error(f"Image Processing Error: {str(e)}")
            return f"Error processing image: {str(e)}"

    def process_documents(self, documents):
        if not documents:
            return None
        texts = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def query(self, qa_chain, question, chat_history):
        response = qa_chain.invoke({"question": question, "chat_history": chat_history})
        return response.get("answer", "No response generated.")

# ✅ Initialize Streamlit Page
st.set_page_config(page_title="AlgebrAI - Math Chatbot", page_icon="🧮", layout="wide")
# Custom CSS for styling chat
st.markdown("""
        <style>
        .user-message {
            background-color: rgb(241 234 26);
            color: black;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            text-align: left;
            float: left;
            clear: both;
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        .ai-message {
            background-color: rgb(163, 168, 184);
            color: black;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            text-align: left;
            float: left;
            clear: both;
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)

st.title("🔢 AlgebrAI - Advanced Math Assistant")
st.write("Ask math questions or upload documents/images for analysis.")

# ✅ Session State Initialization
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = MultiFormatRAG()
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# ✅ Sidebar for File Uploads
with st.sidebar:
    st.title("Upload Files")
    uploaded_files = st.file_uploader("Upload Documents or Images", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv', 'html', 'md', 'png', 'jpg', 'jpeg'])

    if uploaded_files and st.button("Process Files"):
        with st.spinner("Processing..."):
            temp_dir = tempfile.mkdtemp()
            text_data = ""
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                if file.type.startswith('image'):
                    text_data += st.session_state.rag_system.analyze_image(file_path) + "\n"
                else:
                    documents = st.session_state.rag_system.load_documents(temp_dir)
                    if documents:
                        vectorstore = st.session_state.rag_system.process_documents(documents)
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=chat,
                                retriever=vectorstore.as_retriever(),
                                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                            )

            if text_data:
                st.session_state.chat_history.append(AIMessage(content=text_data))
                st.success("Image text extracted successfully!")
            elif st.session_state.qa_chain:
                st.success("Documents processed successfully!")

# ✅ Display Chat History
for msg in st.session_state.chat_history:
    role = "😀" if isinstance(msg, HumanMessage) else "🤖"
    styled_msg = f"""
                <div class="{'user-message' if role == '😀' else 'ai-message'}">
                    <span>{role} : {msg.content}</span>
                </div>
            """
    st.markdown(styled_msg, unsafe_allow_html=True)
# ✅ Chatbot User Input
user_input = st.chat_input("Type your math question...")

if user_input:
    user_message = f"""
            <div class='user-message'>
                <span>😀 : {user_input}</span>
            </div>
        """
    st.markdown(user_message, unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        response = ""
        if st.session_state.qa_chain:
            # Pass chat history to the model
            response = st.session_state.rag_system.query(st.session_state.qa_chain, user_input, st.session_state.chat_history)
        else:
            # Use normal chat without file retrieval
            full_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + st.session_state.chat_history + [HumanMessage(content=user_input)]
            response = chat.invoke(full_prompt).content

    # Append AI response and display it
    st.session_state.chat_history.append(AIMessage(content=response))
    styled_response = f"""
            <div class="ai-message">
                  <span>🤖 : {response}</span>
            </div>
        """
    st.markdown(styled_response, unsafe_allow_html=True)
