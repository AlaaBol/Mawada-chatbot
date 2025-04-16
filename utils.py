from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_ranking_sentence(query: str) -> str:
    prompt = f"""
    You are a helpful assistant for the website Mawada.net. A user asked a question, and your job is to understand their intent and create a clean version of their query that can be used to find the best match from an FAQ database.

    Instructions:
    - Rephrase the question clearly if needed.
    - Keep the core meaning.
    - Focus on matching what the user is asking with similar FAQ entries.
    
    User query: "{query}"
    
    Respond with a rephrased version of the question only. Do not include explanations.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  
        messages=[
            {"role": "system", "content": "You're an assistant that reformulates FAQ search queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()


def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding
