# 🤖 PDF Chatbot with Summarization & Q&A

This project is a smart chatbot that allows users to upload a PDF file and interact with its content through natural language questions. It also provides a 10–15 line summary of the entire document, regardless of its length.

## 🔧 Features

- 📂 Upload any PDF file
- 💬 Ask questions and get answers from the PDF content using semantic search (FAISS + Sentence Transformers)
- 🧠 Summarize the entire PDF into 10–15 key sentences using clustering (KMeans + TF-IDF)
- 🖥️ Clean and simple Gradio UI

## 🛠️ Tech Stack

- Python
- PyMuPDF (for PDF text extraction)
- Sentence Transformers (for embeddings)
- FAISS (for fast similarity search)
- Scikit-learn (TF-IDF + KMeans clustering)
- Gradio (for web interface)

## 🚀 How to Run

1. Clone this repo or download the code.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
