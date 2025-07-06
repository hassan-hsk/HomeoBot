import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import re

# Load environment variable
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
assert groq_key, "Missing GROQ_API_KEY in environment variables."

app = Flask(__name__)
CORS(app)

# Load vectorstore with BioBERT embeddings
embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Setup LLM using Groq
llm = ChatGroq(
    temperature=0,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=groq_key
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# In-memory session store
chat_sessions = {}

# HTML formatting for chatbot responses
def format_response(text):
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\* ([^*\n]+)", r"<li>\1</li>", text)
    text = re.sub(r"\*\*([\w\s]+):\*\*", r"<h4>\1:</h4>", text)
    text = re.sub(r"\n{2,}", r"\n", text)
    text = text.replace("\n", "<br>")
    return text

@app.route("/chat", methods=["POST"])
def chat():
    session_id = request.json.get("session_id")
    user_input = request.json.get("message")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    session = chat_sessions.setdefault(session_id, {"symptoms": [], "rounds": 0})

    # Reset session
    if user_input.lower() == "reset":
        chat_sessions[session_id] = {"symptoms": [], "rounds": 0}
        return jsonify({"response": "🔁 Chat has been reset. Please describe your symptoms."})

    session["symptoms"].append(user_input)

    if session["rounds"] < 4:
        # Homeopathy-specific follow-up prompt:
        query = (
            f"The patient has reported these symptoms: {', '.join(session['symptoms'])}. "
            "Ask a follow-up question that helps to clarify the symptoms in terms of homeopathic principles: "
            "- Ask about modalities (what makes symptoms better or worse), "
            "- sensations, locations, mental/emotional state, and concomitant symptoms. "
            "Do NOT give any diagnosis yet. "
            "Use classical homeopathic language only."
        )
        session["rounds"] += 1
    else:
        # Homeopathy-specific diagnosis + remedy prompt:
        query = (
            f"The patient has reported these symptoms: {', '.join(session['symptoms'])}. "
            "Based on classical homeopathy, provide the most likely diagnosis with homeopathic remedy options. "
            "Include detailed explanation referring to symptoms, modalities, and remedy profiles. "
            "Format your answer with these headings: <strong>Disease</strong>, <strong>Explanation</strong>, and <strong>Medicines</strong>. "
            "Use only homeopathic materia medica and avoid modern medical or allopathic terms."
        )
        session["symptoms"] = []
        session["rounds"] = 0

    response = qa_chain.run(query)
    return jsonify({"response": format_response(response)})

if __name__ == "__main__":
    app.run(debug=True)
