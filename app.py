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
import tempfile
from audio_recorder_streamlit import audio_recorder
import os



print("ClientSettings is working!")

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
        print(f"pdf_chunk_{i}")
        upsert_vector(f"pdf_chunk_{i}", embedding, metadata)

        progress.progress((i + 1) / total)

def generate_answer(query):
    suggestions = []

    # Step 1: Combined classification + language detection + friendly response
    intent_prompt = f"""
    You are a helpful multilingual chatbot assistant for the website Mawada.net.

    Your tasks:
    1. Detect the intent of the user message. It can be one of:
        - greeting (like "hi", "hello", "good morning", "how are you")
        - faq (any question related to Mawada.net services, accounts, subscriptions, or support)
        - off-topic (anything unrelated to Mawada.net)
    
    2. Respond in the **same language** the user used.

    Instructions:
    - If it's a greeting: reply warmly and ask how you can assist in the **same language** the user used..
    - If it's off-topic: respond politely that you can only assist with Mawada.net topics in the **same language** the user used..
    - If it's a valid Mawada.net question: respond with just the word: faq

    Don't explain the classification or your reasoning.
    Only return the response.
    User message:
    \"\"\"{query}\"\"\"
    """

    classification_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": intent_prompt.strip()},
        ],
        temperature=0.3,
        max_tokens=250,
    )

    classification = classification_response.choices[0].message.content.strip().lower()

    # Step 2: If classified as faq, search for best match and polish the result
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
                raw_answer = suggestions[0].split("\n**A:**")[1].strip() if '**A:**' in suggestions[0] else "Sorry, I couldn't find an answer."
                base_response = f"**Q:** {query}\n**A:** {raw_answer}"
            else:
                base_response = "Sorry, I couldn't find an answer. Please rephrase your question or contact support."
        else:
            base_response = "Sorry, I couldn't find an answer. Please rephrase your question or contact support."

        # Step 2.5: Refine FAQ answer into natural language
        final_prompt = f"""
        You are a multilingual assistant for Mawada.net.

        The user asked the following question:
        \"\"\"{query}\"\"\"

        You are given a raw answer (possibly in Arabic). Your task:
        - Rewrite it in a friendly and natural tone.
        - Translate it into the same language the user used in the question.
        - Keep the answer clear and helpful.

        Raw answer:
        \"\"\"{base_response}\"\"\"
        """


        final_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": final_prompt.strip()},
                {"role": "user", "content": query}
            ],
            temperature=1,
            max_tokens=400
        )

        response = final_response.choices[0].message.content.strip()

    else:
        # If not an FAQ, we already have the friendly message from the first call
        response = classification

    # Save conversation
    st.session_state.conversation_history["input"].append(query)
    st.session_state.conversation_history["output"].append(response)
    st.session_state.memory.save_context({"input": query}, {"output": suggestions})

    return response


# UI
st.set_page_config(page_title="Mawada - Chatbot", page_icon=":heart:", layout="wide")

st.markdown("<h1 style='text-align: center;'>♥️ Mawada Ai</h1>", unsafe_allow_html=True)

# Show onboarding popup once per session
if "onboarding_shown" not in st.session_state:
    st.session_state.onboarding_shown = True

    with st.expander("👋 مرحبًا بك في Arabot", expanded=True):
        st.markdown("""
        ### 🤖 ما هو Arabot؟

        Arabot هو مساعد ذكي تم تدريبه لمساعدتك في الإجابة على الأسئلة المتعلقة بمنصة Mawada.net.

        ---
        **ما الذي يمكنه فعله؟**
        - ✅ الإجابة على الأسئلة الشائعة حول الحسابات والاشتراكات والخدمات.
        - ✅ تحليل الملفات (PDF وCSV) واستخراج المعلومات.
        - ✅ التعرف على الأسئلة الصوتية وتحويلها لنصوص والرد عليها.

        **ما الذي لا يمكنه فعله؟**
        - ❌ لا يمكنه تقديم استشارات قانونية أو دينية.
        - ❌ لا يفهم الأسئلة الخارجة عن نطاق Mawada.net.
        - ❌ قد لا تكون بعض الإجابات دقيقة 100%، تأكد دائمًا من المعلومات من المصدر.

        ---
        👂 **نصيحة**: استخدم اللغة العربية لنتائج أدق وأسهل فهمًا!
        """)


# Create columns for input and audio recorder
col1, col2 = st.columns([0.85, 0.15])  # Adjust ratios as needed

with col1:
    query_text = st.text_input(
        "here", 
        placeholder="🤖Ask a Question", 
        label_visibility="collapsed",
        key="main_input"
    )

with col2:
    # Audio Recording Section - inline with input
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone-lines",
        icon_size="2x",
        key="inline_recorder"
    )
# query_text = st.text_input(
#     "here", placeholder="🤖Ask a Question", label_visibility="collapsed"
# )

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
    audio_file = st.file_uploader("Upload your question (WAV or MP3)", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        with st.spinner("Transcribing audio..."):
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            st.success("Transcription complete!")
            st.write("✏️ Transcribed Text:")
            st.write(transcription)

        # Use your existing RAG logic
        with st.spinner("💡 Generating answer..."):
            answer = generate_answer(transcription)
            st.subheader("🤖 AI Answer:")
            st.write(answer)
            col_like, col_dislike = st.columns([0.1, 0.1])
            with col_like:
                if st.button("👍", key=f"thumbs_up_{len(st.session_state.conversation_history['input'])}"):
                    st.success("Thanks for your feedback! 😊")

            with col_dislike:
                if st.button("👎", key=f"thumbs_down_{len(st.session_state.conversation_history['input'])}"):
                    st.warning("Thanks! We’ll use your feedback to improve. 🙏")

# Button to trigger response generation
if st.button("Answer") and query_text:
    with st.spinner("🔎 Searching and Generating Response..."):
        response = generate_answer(query_text)
        st.session_state.last_query = query_text
        st.session_state.last_response = response

# Display response if already generated
if "last_response" in st.session_state:
    response = st.session_state.last_response
    st.subheader("📝 AI Answer:")
    st.write(response)

    response_key = f"{len(st.session_state.conversation_history['input']) - 1}"
    col1, col2, col3 = st.columns([0.1, 0.1, 0.1])

    with col1:
        if st.button("👍", key=f"thumbs_up_{response_key}"):
            st.session_state[f"feedback_{response_key}"] = "up"
    with col2:
        if st.button("👎", key=f"thumbs_down_{response_key}"):
            st.session_state[f"feedback_{response_key}"] = "down"
    with col3:
        if st.button("🚩", key=f"report_{response_key}"):
            st.session_state[f"feedback_{response_key}"] = "report"

    # Show feedback acknowledgment
    feedback = st.session_state.get(f"feedback_{response_key}")
    if feedback == "up":
        st.success("شكراً على تقييمك! 😊")
    elif feedback == "down":
        st.warning("شكراً! سنأخذ ملاحظتك بعين الاعتبار. 🙏")
    elif feedback == "report":
        st.error("تم الإبلاغ عن هذه الإجابة. فريق الدعم سيتحقق منها. 🚨")


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

# Audio Recording Section   

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    with st.spinner("Transcribing audio..."):
        # Save audio bytes to temporary file for OpenAI API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            st.success("Transcription complete!")
            st.write("✏️ Transcribed Text:")
            st.write(transcription)
            
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

        # Automatically generate answer
    with st.spinner("💡 Generating answer..."):
        answer = generate_answer(transcription)
        st.subheader("🤖 AI Answer:")
        st.write(answer)
        # Use a unique key for this message
        
