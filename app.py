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
from pinecone_store import load_csv
from pinecone_store import split_text
from pinecone_store import upsert_vector
from pinecone_store import search
from pinecone_store import load_json_bulk
from pinecone_store import add_single_faq

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

if "conversation_history" not in st.session_state:
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

        progress.progress((i + 1) / total)


# def generate_answer(query):
#     suggestions = []
#     # Ask GPT whether this is a general or FAQ-related question
#     intent_prompt = f"""
#     Classify the user query below as one of the following:
#     - "general": if it's a greeting.
#     - "faq": if it's a question that might relate to Mawada.net's FAQs.

#     Query: "{query}"

#     Respond only with "general" or "faq".
#     """

#     intent_response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "system", "content": "You classify user input into general or faq intent."},
#             {"role": "user", "content": intent_prompt}
#         ],
#         temperature=0,
#         max_tokens=10
#     )

#     intent = intent_response.choices[0].message.content.strip().lower()
#     # print(f"📌 Detected Intent: {intent}")

#     # Handle general conversation via GPT
#     if intent == "general":
#         general_reply_prompt = f"""
#         You are a helpful and friendly assistant for the website Mawada.
#         A user sent the following general message:

#         "{query}"

#         Respond politely and naturally as a helpful assistant might only if it is related to Mawada if not then say sorry I can not help.
#         """

#         reply_response = client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {"role": "system", "content": "You're a polite and helpful chatbot for Mawada.net."},
#                 {"role": "user", "content": general_reply_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=100
#         )

#         response = reply_response.choices[0].message.content.strip()
#         # print(f"💬 General GPT Response: {response}")

#     # Handle FAQ-style questions by embedding + semantic search
#     else:
#         ranking_sentence = extract_ranking_sentence(query)
#         # print(f"🔍 Ranking Sentence: {ranking_sentence}")

#         embeddings = get_embedding(ranking_sentence)
#         result = search(embeddings)

#         if result:
#             # for idx, item in enumerate(result[:3]):  # Adjust 3 for top-k (or any number)
#             #     suggestion = f"**Question:** {item.get('question', 'No question found')}\n> {item.get('answer', 'No answer found')}"
#             for item in result:
#                 metadata, score = item
#                 suggestion = f"**Question:** {metadata.get('question', 'No question found')}\n> {metadata.get('answer', 'No answer found')}"
#                 suggestions.append(suggestion)

#             response = result[0][0].get("answer", "No exact answer found.")
#         else:
#             response = "❌ لم أتمكن من العثور على إجابة تطابق سؤالك. حاول إعادة صياغته أو طرح سؤال آخر."
#         # print(f"📦 FAQ Search Result: {response}")

#     # Save to conversation history and memory
#     st.session_state.conversation_history["input"].append(query)
#     st.session_state.conversation_history["output"].append(response)
#     st.session_state.memory.save_context({"input": query}, {"output": suggestions})

#     return response


# def generate_answer(query):
#     suggestions = []
#     # Step 1: Use chat model to decide how to respond
#     system_prompt = """
#     You are a helpful assistant for the website Mawada.net.

#     When a user sends a message, follow these rules:

#     - If the message is a greeting or small talk like "hi", "hello", "good morning", or "how are you", respond with a polite greeting and ask how you can assist.
#     - If the message is a question or topic related to Mawada.net (services, accounts, subscriptions, support, etc.), respond ONLY with: faq
#     - If the message is unrelated to Mawada.net or is off-topic, respond with: "I'm sorry, I can only help with questions related to Mawada.net."

#     Do not explain or add anything else. Just reply as instructed.
#     """

#     classification_response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "system", "content": system_prompt.strip()},
#             {"role": "user", "content": query},
#         ],
#         temperature=0,
#         max_tokens=50,
#     )

#     classification = classification_response.choices[0].message.content.strip().lower()

#     # Step 2: Handle response
#     if classification == "faq":
#         # Continue with FAQ logic
#         query_embedding = get_embedding(query)
#         result = search(query_embedding)

#         if result:
#             # metadata, score = result[0]
#             for item in result:
#                 metadata, score = item
#                 suggestion = f"**Question:** {metadata.get('question', 'No question found')}\n> {metadata.get('answer', 'No answer found')}"
#                 suggestions.append(suggestion)
#             response = metadata.get("answer", "No exact answer found.")
#         else:
#             response = "Sorry, I couldn't find an answer. Please rephrase your question or contact support."

#     else:
#         # Treat it as a general chatbot reply (greeting)
#         response = (
#             classification  # Should already be a polite greeting with "How can I help?"
#         )
#     st.session_state.conversation_history["input"].append(query)
#     st.session_state.conversation_history["output"].append(response)
#     st.session_state.memory.save_context({"input": query}, {"output": suggestions})
#     return response

def generate_answer(query):
    suggestions = []
    
    # Step 1: Classify the intent
    system_prompt = """
    You are a helpful assistant for the website Mawada.net.

    When a user sends a message, follow these rules:

    - If the message is a greeting or small talk like "hi", "hello", "good morning", or "how are you", respond with a polite greeting and ask how you can assist.
    - If the message is a question or topic related to Mawada.net (services, accounts, subscriptions, support, etc.), respond ONLY with: faq
    - If the message is unrelated to Mawada.net or is off-topic, respond with: "I'm sorry, I can only help with questions related to Mawada.net."

    Do not explain or add anything else. Just reply as instructed.
    """

    classification_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=250,
    )

    classification = classification_response.choices[0].message.content.strip().lower()

    # Step 2: Handle logic based on intent
    if classification == "faq":
        query_embedding = get_embedding(query)
        result = search(query_embedding)

        if result:
            for item in result:
                metadata, score = item
                question = metadata.get('question')
                answer = metadata.get('answer')
                if question and answer:
                    suggestions.append(f"**Q:** {question}\n**A:** {answer}")
            if suggestions:
                answer = suggestions[0].split("\n**A:**")[1].strip() if '**A:**' in suggestions[0] else "Sorry, I couldn't find an answer."
                base_response =f"**Q:** {query}\n**A:** {answer}" 
        else:
            base_response =""
    
    else:
        base_response = classification 
        
# Check if base_response is empty or whitespace
    if not base_response.strip():
        response = "Sorry, I couldn't find an answer. Please rephrase your question or contact support."
    else:
        # Finalize with chat model for a smooth, human-friendly reply
        final_prompt = f"""
     Refine the message below into a natural, helpful response.

    Original response:
    {base_response}

    Do not explain the instructions again. The response should be polished and friendly, preserving all the important details. If it is an FAQ, organize and make it clear. If it's a greeting or an apology, just make it friendlier.

    """

        final_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You refine assistant messages."},
                {"role": "user", "content": final_prompt.strip()}
            ],
            temperature=0.5,
            max_tokens=400
        )

        response = final_response.choices[0].message.content.strip()

    # Save conversation
    st.session_state.conversation_history["input"].append(query)
    st.session_state.conversation_history["output"].append(response)
    st.session_state.memory.save_context({"input": query}, {"output": suggestions})

    return response



# UI
st.set_page_config(page_title="Info Extraction", page_icon=":books:", layout="wide")

st.markdown("<h1 style='text-align: center;'>🔍 Mawada Ai</h1>", unsafe_allow_html=True)

query_text = st.text_input(
    "here", placeholder="📖 Ask a Question", label_visibility="collapsed"
)

with st.sidebar:
    st.subheader("📂 My Documents")
    uploaded_files = st.file_uploader(
        "PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDF Documents"):
        with st.progress(0):
            process_pdf_documents(uploaded_files)

    uploaded_csv_files = st.file_uploader(
        "Upload CSV files", type="csv", accept_multiple_files=True
    )

    if uploaded_csv_files and st.button("Process CSV Files"):
        with st.progress(0):
            load_csv(uploaded_csv_files)

    # st.markdown("---")
    # st.subheader("➕ Add FAQ")

    # input_question = st.text_input("Question", key="faq_q")
    # input_answer = st.text_area("Answer", key="faq_a")

    # if st.button("Add FAQ"):
    #     if input_question.strip() and input_answer.strip():
    #         add_single_faq(input_question, input_answer)
    #         st.success("FAQ added successfully!")
    #     else:
    #         st.warning("Please fill both question and answer.")

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
