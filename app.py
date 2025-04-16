
import streamlit as st
import os
from langchain.schema import HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModel
import subprocess
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from openai import OpenAI
from utils import get_embedding
from utils import extract_ranking_sentence
from pinecone_store import get_pdf_text
from pinecone_store import split_text
from pinecone_store import upsert_vector
from pinecone_store import search
from pinecone_store import load_json_bulk

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = {"input": [], "output": []}
    
def process_pdf_documents(uploaded_files):
    """Process PDFs and store embeddings in Pinecone."""
    global index, document_chunks
    
    text = get_pdf_text(uploaded_files)
    chunks = split_text(text)
    document_chunks = chunks  
    
    total = len(document_chunks)
    progress = st.progress(0)

    for i, chunk in enumerate(document_chunks):
        metadata = {"source": "pdf", "chunk": i, "text": chunk}
        embedding = get_embedding(chunk)
        upsert_vector(f"pdf_chunk_{i}", embedding, metadata)

        # Smooth progress update
        progress.progress((i + 1) / total)

    st.sidebar.success(f"✅ Processed {len(uploaded_files)} pdf document(s) with {total} chunks.")

# def generate_answer(query):
#     """Retrieve relevant text and generate an answer"""
#     print(f"🔍 User Query: {query}")
#     ranking_sentence = extract_ranking_sentence(query)
#     print(f"📌 Extracted Ranking Sentence: {ranking_sentence}")
#     embeddings = get_embedding(ranking_sentence)
#     print(f"📊 Embedding (first 5 values): {embeddings[:2]}")
#     # Get  response
#     result = search(embeddings)
#     print(f"📦 Search Results: {result}")
#     st.session_state.conversation_history["input"].append(query)
#     st.session_state.conversation_history["output"].append(result)

#     # Save context to memory
#     st.session_state.memory.save_context({"input": query}, {"output": result}) 
#         # Return first answer or fallback
#     if result:
#         return result[0].get("answer", "No exact answer found.")
#     else:
#         return "❌ No corresponding answer found."

def generate_answer(query):
    suggestions = []

    # print(f"🧠 Incoming Query: {query}")

    # Ask GPT whether this is a general or FAQ-related question
    intent_prompt = f"""
    Classify the user query below as one of the following:
    - "general": if it's a greeting.
    - "faq": if it's a question that might relate to Mawada.net's FAQs.

    Query: "{query}"

    Respond only with "general" or "faq".
    """

    intent_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You classify user input into general or faq intent."},
            {"role": "user", "content": intent_prompt}
        ],
        temperature=0,
        max_tokens=10
    )

    intent = intent_response.choices[0].message.content.strip().lower()
    # print(f"📌 Detected Intent: {intent}")

    # Handle general conversation via GPT
    if intent == "general":
        general_reply_prompt = f"""
        You are a helpful and friendly assistant for the website Mawada.
        A user sent the following general message:

        "{query}"

        Respond politely and naturally as a helpful assistant might only if it is related to Mawada مودة if not then say sorry I can not help.
        """

        reply_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You're a polite and helpful chatbot for Mawada.net."},
                {"role": "user", "content": general_reply_prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )

        response = reply_response.choices[0].message.content.strip()
        # print(f"💬 General GPT Response: {response}")

    # Handle FAQ-style questions by embedding + semantic search
    else:
        ranking_sentence = extract_ranking_sentence(query)
        # print(f"🔍 Ranking Sentence: {ranking_sentence}")

        embeddings = get_embedding(ranking_sentence)
        result = search(embeddings)
        
        if result:
            # for idx, item in enumerate(result[:3]):  # Adjust 3 for top-k (or any number)
            #     st.markdown(f"**Suggestion {idx + 1}:**")
            #     st.markdown(f"**Question:** {item.get('question', 'No question found')}")
            #     st.markdown(f"> {item.get('answer', 'No answer found')}")
            for idx, item in enumerate(result[:3]):  # Adjust 3 for top-k (or any number)
                suggestion = f"**Question:** {item.get('question', 'No question found')}\n> {item.get('answer', 'No answer found')}"
                suggestions.append(suggestion)

            response = result[0].get("answer", "No exact answer found.")
        else:
            response = "❌ لم أتمكن من العثور على إجابة تطابق سؤالك. حاول إعادة صياغته أو طرح سؤال آخر."
        # print(f"📦 FAQ Search Result: {response}")

    # Save to conversation history and memory
    st.session_state.conversation_history["input"].append(query)
    st.session_state.conversation_history["output"].append(response)
    st.session_state.memory.save_context({"input": query}, {"output": suggestions})

    return response


# Streamlit UI
st.set_page_config(page_title="Info Extraction", page_icon=":books:", layout="wide")

st.markdown("<h1 style='text-align: center;'>🔍 Mawada Ai</h1>", unsafe_allow_html=True)

query_text = st.text_input("here", placeholder="📖 Ask a Question",label_visibility="collapsed")

with st.sidebar:
    st.subheader("📂 My Documents")
    uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process PDF Documents"):
        with st.progress(0):
            process_pdf_documents(uploaded_files)

if query_text and st.button("Answer"):
    with st.spinner("🔎 Searching and Generating Response..."):
        response = generate_answer(query_text)

        # 🔽 Show results
        # if isinstance(response, list):
        #     for i, item in enumerate(response):
        #         st.markdown(f"**Suggestion {i+1}:** {item.get('question', '')}")
        #         st.markdown(f"> {item.get('answer', '')}")
        # else:
        #     st.warning(response)

    if response:
        st.subheader("📝 AI Answer:")
        st.write(response)
        
    else:
        st.warning("No response generated.")
        

    with st.expander("💬 Conversation History:"):
        chat_data = st.session_state.memory.load_memory_variables({})
        chat_history = chat_data.get("chat_history", [])

        if chat_history:
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    st.write(f"**Q:** {message.content}")  
                elif isinstance(message, AIMessage):
                    st.write(f"**A:** {message.content}") 
        else:
            st.write("No Conversation History is Available")

 

