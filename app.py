import fitz  # PyMuPDF
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text
def extract_text_from_pdf(pdf_file):
    file_path = pdf_file.name
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Build FAISS index
def build_index(text):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return sentences, embeddings, index

# Answer a query
def answer_question(pdf_file, query):
    text = extract_text_from_pdf(pdf_file)
    sentences, embeddings, index = build_index(text)
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    return sentences[I[0][0]]

# Summarize PDF
def summarize_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # If fewer than 10 sentences, return them all
    if len(sentences) <= 10:
        return '\n'.join(sentences)

    # If more, apply clustering for best 10
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    n_clusters = min(10, len(sentences))  # avoid error
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_centers = kmeans.cluster_centers_

    summary_sentences = []
    for i in range(n_clusters):
        idx = np.argmin(np.linalg.norm(X.toarray() - cluster_centers[i], axis=1))
        summary_sentences.append(sentences[idx])

    # Sort summary by original order in the document
    summary_sentences = sorted(summary_sentences, key=lambda s: sentences.index(s))

    return '\n'.join(summary_sentences)

def summarize_text_input(user_text):
    # Break the input into sentences
    sentences = user_text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= 2:
        return "Please provide more detailed text (at least 2-3 sentences) to summarize."

    # Get sentence embeddings
    embeddings = model.encode(sentences)
    
    # Use FAISS to find most central sentences (summary-like)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    D, I = index.search(np.array([np.mean(embeddings, axis=0)]), k=min(10, len(sentences)))

    # Return top representative sentences
    summary = "\n".join([sentences[i] for i in I[0]])
    return summary



# UI
with gr.Blocks(title="PDF Smart Chatbot") as demo:
    gr.Markdown("## 📄 PDF Chatbot with Q&A + Summarizer")

    with gr.Row():
        pdf_input = gr.File(label="📂 Upload your PDF")
    with gr.Row():
        question = gr.Textbox(label="🔍 Ask a question based on the PDF")
        answer_btn = gr.Button("💬 Get Answer")
        summary_btn = gr.Button("🧠 Summarize PDF")

    output = gr.Textbox(label="📌 Result", lines=15)

    answer_btn.click(fn=answer_question, inputs=[pdf_input, question], outputs=output)
    summary_btn.click(fn=summarize_pdf, inputs=[pdf_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
