import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
from utils import get_embedding
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
import io
load_dotenv()

# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Connect to  index
index = pc.Index(os.environ.get("PINECONE_INDEX"))

def upsert_vector(id: str, embedding: list, metadata: dict):
    index.upsert([(id, embedding, metadata)])

def search(embedding: list, top_k: int = 3):
    # res = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    # return [match['metadata'] for match in res['matches']]
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,  # Optional, you might not need full vector back
    )
    
    filtered_matches = []
    for match in result.matches:
        score = match.score  
        print("score",score)
        if score >= 0.50:
            filtered_matches.append((match.metadata, score))

    return filtered_matches

# Load multiple JSON files from 'faq_data' folder
def load_json_bulk(folder_path: str = "zendesk"):
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    for file_name in file_list:
        with open(os.path.join(folder_path, file_name)) as f:
            faq_data = json.load(f)
            for i, item in enumerate(faq_data):
                combined_text = f"Q: {item['question']}\nA: {item['answer']}"
                embedding = get_embedding(combined_text)
                unique_id = f"{file_name}_{i}"
                upsert_vector(unique_id, embedding, item)

# Load csv file
def load_csv(file_list):
    for file in file_list:
        content = file.read().decode("utf-8")
        file.seek(0)

        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)  
        total = len(rows)

        progress = st.progress(0)

        for i, row in enumerate(rows):
            combined_text = f"Q: {row['question']}\nA: {row['answer']}"
            print(combined_text)

            embedding = get_embedding(combined_text)
            upsert_vector(f"{file.name}_{i}", embedding, row)

            progress.progress((i + 1) / total) 


#load pdfs
def get_pdf_text(files):
    """Extract text from uploaded PDFs."""
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text(text):
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_text(text)
