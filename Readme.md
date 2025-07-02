# 📄 Chat with Your PDFs

live web : https://chethan616-chatwithpdf-app-sd2s5u.streamlit.app

https://github.com/Chethan616/ChatWithPDF/blob/main/Screenshot_2025-07-02-18-27-39-35_40deb401b9ffe8e1df2f1cc5ba480b12.jpg

An AI-powered Streamlit web app that allows you to upload PDF documents and chat with them using natural language. It leverages **Google Gemini models** and **FAISS vector stores** to understand and answer questions based on your PDF content.

---

## 🚀 Features

- 📄 Upload **multiple PDF files**
- 🤖 Chat with your documents using **Gemini 1.5 Flash**
- 🔍 Retrieves and answers based on **semantic search**
- 🧠 Maintains chat memory with **conversation history**
- ⚡ Uses **FAISS** for fast and scalable vector storage
- 🌐 Built with **Streamlit** and powered by **LangChain**

---

## 🧠 How It Works

1. **Upload PDFs** in the sidebar

2. The app:
   - 📄 Extracts text using `PyPDF2`
   - ✂️ Splits it into manageable chunks
   - 🧠 Converts them into embeddings via `GoogleGenerativeAIEmbeddings`
   - 🗃️ Stores them in a **FAISS** vector store
   - 🤖 Builds a **ConversationalRetrievalChain** using `ChatGoogleGenerativeAI`

---

## 📦 Requirements

```bash
streamlit
langchain
langchain-google-genai
faiss-cpu
PyPDF2
python-dotenv


pip install -r requirements.txt

GOOGLE_API_KEY=your_google_api_key_here

streamlit run app.py
