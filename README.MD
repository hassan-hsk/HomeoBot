
# 🧠 Homeopathic AI Doctor

A web-based AI assistant that interacts with patients, analyzes their symptoms, and recommends homeopathic remedies using natural language processing and embedded medical knowledge.

---

## 📌 Features

- 🤖 **Conversational Chatbot** powered by LLMs (LLaMA via Groq API)
- 📚 **Symptom-to-Remedy Matching** using pre-embedded homeopathic text data
- 🔄 **Contextual Chat Memory** to guide follow-up questions before diagnosis
- 🌐 **Simple Web Frontend** built with HTML + JavaScript
- 💾 **Efficient Vector Embedding** stored with ChromaDB to avoid reprocessing on restart

---

## 🏗️ Tech Stack

- **Frontend:** HTML, CSS, Vanilla JS  
- **Backend:** Python (Flask), LangChain, HuggingFace Embeddings, ChromaDB  
- **LLM API:** [Groq's LLaMA-4 Scout](https://groq.com/)  
- **Deployment-ready** for Render/Vercel/Localhost

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hassanawi/homeo-ai-doctor.git
cd homeo-ai-doctor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Generate Embeddings (Only Once)

```bash
python embedding_setup.py
```

### 5. Run the App

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧪 Example Symptoms to Test

- "Swelling in legs, aching pain, better with elevation"
- "Sore throat, yellow discharge, worse at night"
- "Sharp stomach pain, craving cold drinks, worse at 3 a.m."

---

## 📁 Project Structure

```
├── app.py                   # Flask backend
├── embedding_setup.py       # Embedding generator (run once)
├── homeo_medicine.txt       # Homeopathy data source
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not included)
└── index.html               # Frontend UI
```

---

## 📄 License

This project is for educational and experimental purposes. Not intended to replace professional medical advice.
