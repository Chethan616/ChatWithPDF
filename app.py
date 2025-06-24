import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from typing import List

# --- HELPER FUNCTIONS ---

def get_pdf_text(pdf_docs: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits long text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks: List[str]):
    """Creates a FAISS vector store using Google embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not text_chunks:
        return None
    try:
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_conversation_chain(vectorstore):
    """Creates the conversational retrieval chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

# --- MAIN APP ---

def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🚨 Google API Key not found. Please set it in your .env file or Streamlit Secrets.")
        st.stop()

    st.set_page_config(page_title="Chat with Your PDFs", page_icon="📄", layout="wide")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    # --- SIDEBAR FILE UPLOAD ---
    with st.sidebar:
        st.header("📄 Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process Documents"):
            if pdf_docs:
                with st.status("Processing documents...", expanded=True) as status:
                    st.write("1. Extracting text from PDFs...")
                    raw_text = get_pdf_text(pdf_docs)
                    status.update(label="1. Text extracted.")

                    if raw_text:
                        st.write("2. Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        status.update(label="2. Text chunked.")

                        st.write("3. Creating vector store...")
                        vectorstore = get_vectorstore(text_chunks)
                        status.update(label="3. Vector store created.")

                        if vectorstore:
                            st.write("4. Building conversation chain...")
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.session_state.processing_done = True
                            st.session_state.messages = []
                            status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                        else:
                            status.update(label="⚠️ Error in vector store creation.", state="error")
                    else:
                        status.update(label="⚠️ No text could be extracted from the PDFs.", state="error")
            else:
                st.warning("Please upload at least one PDF file.")

    # --- MAIN CHAT INTERFACE ---
    st.title("🤖 Chat with Your PDFs")
    st.info("Upload your documents in the sidebar and click 'Process' to begin.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.processing_done):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = {}
                try:
                    response = st.session_state.conversation({'question': prompt})
                    answer = response.get('answer', "Sorry, I couldn't find an answer.")
                except Exception as e:
                    st.error(f"⚠️ Internal Error: {e}")
                    answer = "Error during processing."

                st.markdown(answer)

                with st.expander("View Sources"):
                    if 'source_documents' in response and response['source_documents']:
                        for doc in response['source_documents']:
                            st.write("**Source:**")
                            st.info(f"{doc.page_content[:250]}...")
                    else:
                        st.write("No source documents found.")

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == '__main__':
    main()
