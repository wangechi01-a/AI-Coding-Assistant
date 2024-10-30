import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain_community.vectorstores import FAISS  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain.chains import RetrievalQA
from config import OPENAI_API_KEY, PERSIST_DIRECTORY, FAISS_SETTINGS
import json
import pickle
from langchain_core.runnables import RunnableLambda
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_core.documents import Document 

# Load the LLM model
checkpoint = "MBZUAI/LaMini-T5-738M"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_faiss_index_exists(faiss_directory):
    index_file_path = os.path.join(faiss_directory, "index.faiss")
    return os.path.exists(index_file_path)

def load_db():
    # Ensure the FAISS directory exists
    faiss_directory = FAISS_SETTINGS["persist_directory"]
    ensure_directory_exists(faiss_directory)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sample_text = "Sample text to determine embedding dimensions."
    embed_vector = embeddings.embed_query(sample_text)
    embed_dim = len(embed_vector)

    if not ensure_faiss_index_exists(faiss_directory):
        index = faiss.IndexFlatL2(embed_dim)
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

        def embedding_function(texts):
            return embeddings.embed_query(texts)

        db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_function)
        db.save_local(faiss_directory)
        print("New FAISS index created and saved.")
    else:
        db = FAISS.load_local(faiss_directory, embeddings, allow_dangerous_deserialization=True)

    return db

@st.cache_resource
def llm_pipeline():
    hf_pipeline = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    return RunnableLambda(lambda x, **kwargs: hf_pipeline(x)[0]['generated_text'].strip())

def process_query(query):
    db = load_db()
    retriever = db.as_retriever(search_type="similarity")
    
    # Create a retrieval chain without the return_only_output parameter
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline(), 
        retriever=retriever
    )
    
    try:
        result = qa_chain.invoke({"query": query})
        print(f"Query Result: {result}")  # Debug print to inspect the result structure

        # Handle different output types robustly
        if isinstance(result, str):
            return result.strip()  # If the result is already a string
        elif isinstance(result, dict):
            return result.get('output', 'I don\'t know the answer to that question.').strip()
        else:
            return 'I don\'t know the answer to that question.'
    except Exception as e:
        error_message = str(e).replace('\n', ' ').replace('\\n', ' ')
        if 'Helpful Answer:' in error_message:
            cleaned_message = error_message.split('Helpful Answer:')[-1].strip()
            return f"Answer: {cleaned_message}"
        else:
            return f"An error occurred: {error_message}"

# Ingest data into the vector store
def ingest_data(data):
    try:
        if isinstance(data, str) and data.endswith('.pkl'):
            with open(data, 'rb') as f:
                data_to_ingest = pickle.load(f, allow_dangerous_deserialization=True)
        else:
            data_to_ingest = data
        
        db = load_db()
        texts = [Document(page_content=str(text)) for text in data_to_ingest]
        db.add_documents(texts)
        db.save_local(FAISS_SETTINGS["persist_directory"])
        print("Data ingested successfully.")
    except Exception as e:
        print(f"Error ingesting data: {e}")

def main():
    st.title("AI Coding Assistant")

    user_query = st.text_area("Enter your query or code problem you are encountering:")

    if user_query:
        result = process_query(user_query)
        st.write(result)

    ingest_text = st.text_area("Enter data to ingest:")
    if st.button("Ingest Data"):
        if ingest_text:
            try:
                data_to_ingest = json.loads(ingest_text)
                if isinstance(data_to_ingest, list):
                    ingest_data(data_to_ingest)
                    st.success("Data ingested successfully.")
                else:
                    st.error("Input data must be a JSON array.")
            except json.JSONDecodeError:
                data_to_ingest = [item.strip() for item in ingest_text.split(",")]
                ingest_data(data_to_ingest)
                st.success("Data ingested successfully.")
            except Exception as e:
                st.error(f"Error ingesting data: {e}")
        else:
            st.error("Please enter some data to ingest.")

if __name__ == "__main__":
    main()
