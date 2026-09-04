🏋️ AI Fitness & Nutrition Assistant

A Hybrid Retrieval-Augmented Generation (RAG) application that
answers fitness, workout, diet, nutrition, and calorie-related questions
using fitness and nutrition PDF documents.

The project combines ChromaDB, BM25, Reciprocal Rank Fusion (RRF),
Hugging Face embeddings, DeepSeek-V3, Streamlit, and Ragas for
retrieval, generation, and evaluation.

🚀 Features

📄 PDF document ingestion

🔎 Semantic/Dense Retrieval using ChromaDB

🔤 Keyword Retrieval using BM25

🔀 Hybrid Retrieval using Reciprocal Rank Fusion (RRF)

🤖 DeepSeek-V3 for answer generation through Hugging Face

🌍 Multi-language responses

💬 Streamlit chat interface

📊 Ragas evaluation

📈 RAG quality metrics:

Faithfulness

Answer Relevancy

Context Precision

Context Recall

⬇️ Download chat history as JSON

🏗️ Architecture

                 Fitness PDF
                     │
                     ▼
              Document Loading
                     │
                     ▼
             Text Chunking
                     │
                     ▼
              Hugging Face
                Embeddings
                     │
                     ▼
                ChromaDB
                     │
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Dense Retrieval            BM25 Retrieval
        │                         │
        └────────────┬────────────┘
                     ▼
              Hybrid Retrieval
                     │
                     ▼
             Reciprocal Rank
                Fusion
                     │
                     ▼
               DeepSeek-V3
                     │
                     ▼
                  Answer
                     │
                     ▼
                Streamlit
                     │
                     ▼
             Ragas Evaluation
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
 Faithfulness   Answer Relevancy   Context
                              Precision / Recall

📁 Project Structure

RAG-PROJECT-1/
│
├── files/
│   ├── app.py
│   ├── create_db.py
│   ├── rag_system.py
│   ├── evaluation.py
│   ├── requirements.txt
│   ├── README.md
│   ├── .gitignore
│   │
│   ├── fitness_chroma_db/
│   └── nutrition_chroma_db/
│
├── fitbook.pdf
└── NutritiveValueofFoods-merged.pdf

🔧 Technologies Used

Technology              Purpose

Python                  Core programming
LangChain               RAG components
ChromaDB                Vector database
BM25                    Keyword retrieval
Hugging Face            Embeddings and LLM access
DeepSeek-V3             Answer generation
Sentence Transformers   Text embeddings
Ragas                   RAG evaluation
Streamlit               Web application

📄 Documents

The application uses two main PDF sources:

fitbook.pdf --- fitness and workout information

NutritiveValueofFoods-merged.pdf --- nutrition and food
information

🔎 Hybrid Retrieval

The project combines semantic search and keyword search.

Dense Retrieval

ChromaDB retrieves semantically similar chunks using:

sentence-transformers/all-MiniLM-L6-v2

BM25 Retrieval

BM25 performs keyword-based retrieval.

Reciprocal Rank Fusion

Results from both retrievers are combined using:

RRF Score = 1 / (60 + rank)

This helps retrieve documents using both semantic meaning and exact
keywords.

🤖 Answer Generation

The retrieved context is passed to DeepSeek-V3 through Hugging Face.

The model is instructed to answer using the retrieved document context,
helping keep responses grounded in the source documents.

📊 Ragas Evaluation

The project uses Ragas to evaluate RAG quality.

1. Faithfulness

Measures whether the generated answer is supported by the retrieved
context.

2. Answer Relevancy

Measures how relevant the generated answer is to the user's question.

3. Context Precision

Measures whether the retrieved contexts are relevant to the question.

4. Context Recall

Measures whether the retrieved contexts contain the information required
to answer the question.

Scores are generally between:

0 → Poor
1 → Better

▶️ Installation

Create a virtual environment:

python -m venv rag

Activate it on Windows:

rag\Scripts\activate

Install dependencies:

pip install -r files/requirements.txt

🔐 Environment Variables

Create a .env file inside the files directory:

hug=YOUR_HUGGINGFACE_TOKEN

Do not upload .env to GitHub.

Add the following to .gitignore:

.env
rag/
__pycache__/
*.pyc
fitness_chroma_db/
nutrition_chroma_db/

🗄️ Create ChromaDB

From the files directory:

python create_db.py

This creates the ChromaDB collections used by the application.

💬 Run the Streamlit Application

From the files directory:

streamlit run app.py

The application will open in your browser.

📊 Run Ragas Evaluation

Run:

python evaluation.py

The evaluation checks:

Faithfulness
Answer Relevancy
Context Precision
Context Recall

The Streamlit application also provides an Evaluate Answer option
for evaluating the latest generated response.

🧪 Example

Question

What exercises train shoulders?

Answer

1. Standing Cable Lateral Raise
2. Converging Shoulder Press

The answer is generated using the retrieved fitness document context.

🔄 Complete Workflow

User Question
      ↓
Topic Detection
      ↓
Fitness / Nutrition / Both
      ↓
Dense Retrieval
      +
BM25 Retrieval
      ↓
Reciprocal Rank Fusion
      ↓
Top Relevant Documents
      ↓
DeepSeek-V3
      ↓
Grounded Answer
      ↓
Ragas Evaluation
      ↓
Quality Metrics

🔒 Security

Never commit:

.env
API tokens
Hugging Face tokens
private credentials

Use environment variables for sensitive credentials.

👨‍💻 Project Purpose

This project demonstrates a practical Hybrid RAG system using:

Vector search

Keyword search

Hybrid retrieval

Reciprocal Rank Fusion

LLM generation

ChromaDB

Hugging Face

Ragas evaluation

Streamlit

It is designed as a portfolio and interview project demonstrating
practical Generative AI and RAG engineering skills.