import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------- Get Project Paths ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

FITBOOK_PATH = os.path.join(
    PROJECT_ROOT,
    "fitbook.pdf"
)

FOODBOOK_PATH = os.path.join(
    PROJECT_ROOT,
    "NutritiveValueofFoods-merged.pdf"
)


# ---------- Load PDFs ----------

print("Loading PDFs...")

fit_loader = PyPDFLoader(FITBOOK_PATH)
fitness_docs = fit_loader.load()

food_loader = PyPDFLoader(FOODBOOK_PATH)
nutrition_docs = food_loader.load()

print("PDF Loaded Successfully")


# ---------- Chunking ----------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

fitness_chunks = text_splitter.split_documents(
    fitness_docs
)

nutrition_chunks = text_splitter.split_documents(
    nutrition_docs
)

print("Fitness chunks:", len(fitness_chunks))
print("Nutrition chunks:", len(nutrition_chunks))


# ---------- Embeddings ----------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings Model Loaded")


# ---------- Create Fitness ChromaDB ----------

fitness_db = Chroma.from_documents(
    documents=fitness_chunks,
    embedding=embeddings,
    collection_name="fitness",
    persist_directory=os.path.join(
        BASE_DIR,
        "fitness_chroma_db"
    )
)

print("Fitness ChromaDB Created")


# ---------- Create Nutrition ChromaDB ----------

nutrition_db = Chroma.from_documents(
    documents=nutrition_chunks,
    embedding=embeddings,
    collection_name="nutrition",
    persist_directory=os.path.join(
        BASE_DIR,
        "nutrition_chroma_db"
    )
)

print("Nutrition ChromaDB Created")


print("All databases created successfully")