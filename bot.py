import faiss
import pandas as pd
import PyPDF2
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and process PDF knowledge base
def load_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts

# Load and process CSV patient database
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Create FAISS index
def create_faiss_index(texts):
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, texts

# Retrieve relevant text from FAISS
def retrieve_faiss(query, index, texts, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [texts[idx] for idx in indices[0]]
    return results

# Retrieve patient vitals from CSV
def get_patient_vitals(query, patient_data):
    if patient_data is not None:
        matching_rows = patient_data[patient_data.apply(lambda row: query.lower() in row.to_string().lower(), axis=1)]
        return matching_rows.to_dict(orient='records') if not matching_rows.empty else []
    return []

# Generate response ussing deepsek
def generate_response(context, query, patient_info):
    patient_context = f"\nPatient Vitals: {patient_info}" if patient_info else ""
    prompt = (
        "You are an AI-powered ICU monitoring assistant specializing in early warning system (EWS) detection, "
        "including NEWS (National Early Warning Score) assessment. Your task is to analyze real-time ICU patient vitals, "
        "identify critical conditions such as bradycardia, tachycardia, hypoxia, and sepsis risk, and provide actionable alerts "
        "to doctors and medical staff. \n\n"
        f"Context: {context}{patient_context}\nUser Query: {query}\nResponse:"
    )
    response = ollama.generate(model='deepseek-r1', prompt=prompt)
    return response['response']

# Streamlit UI
def main():
    st.title("Medical Nurse Assistant Chatbot")
    st.sidebar.header("Upload Data")
    pdf_files = st.sidebar.file_uploader("Upload PDF Knowledge Base", accept_multiple_files=True, type=["pdf"])
    csv_file = st.sidebar.file_uploader("Upload Patient Vitals Database", type=["csv"])
    
    if st.sidebar.button("Process Data"):
        if pdf_files:
            pdf_texts = load_pdfs([file.name for file in pdf_files])
            faiss_index, text_corpus = create_faiss_index(pdf_texts)
            st.session_state["faiss_index"] = faiss_index
            st.session_state["text_corpus"] = text_corpus
            st.success("PDF Knowledge Base Processed Successfully!")
        if csv_file:
            patient_data = load_csv(csv_file.name)
            st.session_state["patient_data"] = patient_data
            st.success("CSV Patient Vitals Database Processed Successfully!")
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    query = st.text_input("Ask a medical question:")
    if st.button("Get Response") and query:
        if "faiss_index" in st.session_state and "text_corpus" in st.session_state:
            retrieved_texts = retrieve_faiss(query, st.session_state["faiss_index"], st.session_state["text_corpus"])
            context = "\n".join(retrieved_texts)
            patient_info = get_patient_vitals(query, st.session_state.get("patient_data"))
            response = generate_response(context, query, patient_info)
            st.session_state["chat_history"].append({"query": query, "response": response})
            
            for chat in st.session_state["chat_history"]:
                st.write(f"**User:** {chat['query']}")
                st.write(f"**AI:** {chat['response']}")
        else:
            st.warning("Please upload and process the data first!")

if __name__ == "__main__":
    main()
