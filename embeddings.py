import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

# Load and process documents
documents = []
for file_path in glob.glob("data/*.txt"):
    loader = TextLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path  # Add source metadata
        documents.append(doc)

print(f"📄 Loaded {len(documents)} documents")

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(  
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
splits = text_splitter.split_documents(documents)
print("📄 splitted documents")

# Use medical-specific sentence transformer for embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
) 

print("loaded model")

# Create and persist vectorstore
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ Embeddings created and stored in ./chroma_db")
