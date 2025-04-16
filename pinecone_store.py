import os
from pinecone import Pinecone
from dotenv import load_dotenv
from utils import get_embedding
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
load_dotenv()

# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Connect to  index
index = pc.Index(os.environ.get("PINECONE_INDEX"))

def upsert_vector(id: str, embedding: list, metadata: dict):
    index.upsert([(id, embedding, metadata)])

def search(embedding: list, top_k: int = 5):
    res = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return [match['metadata'] for match in res['matches']]

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
def load_csv(file_path: str):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            combined_text = f"Q: {row['question']}\nA: {row['answer']}"
            embedding = get_embedding(combined_text)
            upsert_vector(f"csv_{i}", embedding, row)

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
