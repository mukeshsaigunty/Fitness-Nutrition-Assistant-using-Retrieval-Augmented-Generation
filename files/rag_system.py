import os
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("hug")


# ============================================================
# IMPORTS
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from huggingface_hub import InferenceClient


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD CHROMA DATABASES
# ============================================================

fitness_db = Chroma(
    collection_name="fitness",
    embedding_function=embeddings,
    persist_directory=os.path.join(
        BASE_DIR,
        "fitness_chroma_db"
    )
)


nutrition_db = Chroma(
    collection_name="nutrition",
    embedding_function=embeddings,
    persist_directory=os.path.join(
        BASE_DIR,
        "nutrition_chroma_db"
    )
)


print("ChromaDB Loaded Successfully")


# ============================================================
# GET DOCUMENTS FROM CHROMA
# ============================================================

def get_chroma_documents(vector_db):

    data = vector_db.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = []

    for i, content in enumerate(
        data["documents"]
    ):

        metadata = {}

        if data.get("metadatas"):
            metadata = data["metadatas"][i] or {}

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return documents


# ============================================================
# LOAD DOCUMENTS FOR BM25
# ============================================================

fitness_documents = get_chroma_documents(
    fitness_db
)

nutrition_documents = get_chroma_documents(
    nutrition_db
)


# ============================================================
# BM25 RETRIEVERS
# ============================================================

fitness_bm25 = BM25Retriever.from_documents(
    fitness_documents
)

fitness_bm25.k = 3


nutrition_bm25 = BM25Retriever.from_documents(
    nutrition_documents
)

nutrition_bm25.k = 3


print("BM25 Retrievers Loaded Successfully")


# ============================================================
# DEEPSEEK MODEL
# ============================================================

client = InferenceClient(
    model="deepseek-ai/DeepSeek-V3",
    token=HF_TOKEN
)


# ============================================================
# QUERY ROUTER
# ============================================================

def detect_topic(query):

    fitness_keywords = [
        "exercise",
        "workout",
        "gym",
        "muscle",
        "training"
    ]

    food_keywords = [
        "food",
        "diet",
        "nutrition",
        "vitamin",
        "calories",
        "protein",
        "fat"
    ]

    query = query.lower()

    fitness = any(
        word in query
        for word in fitness_keywords
    )

    food = any(
        word in query
        for word in food_keywords
    )

    if fitness and food:
        return "both"

    elif fitness:
        return "fitness"

    elif food:
        return "nutrition"

    else:
        return "both"


# ============================================================
# DENSE RETRIEVAL WITH SCORES
# ============================================================

def dense_retrieval(
    query,
    vector_db,
    k=3
):

    results = vector_db.similarity_search_with_relevance_scores(
        query,
        k=k
    )

    return results


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieval(
    query,
    vector_db,
    bm25_retriever,
    dense_k=3,
    sparse_k=3,
    final_k=4
):

    # --------------------------------------------------------
    # DENSE RETRIEVAL
    # --------------------------------------------------------

    dense_results = dense_retrieval(
        query,
        vector_db,
        dense_k
    )

    dense_docs = [
        doc
        for doc, score in dense_results
    ]


    # --------------------------------------------------------
    # CHECK DENSE RELEVANCE
    # --------------------------------------------------------

    dense_scores = [
        score
        for doc, score in dense_results
    ]

    max_dense_score = (
        max(dense_scores)
        if dense_scores
        else 0
    )


    # --------------------------------------------------------
    # SPARSE RETRIEVAL - BM25
    # --------------------------------------------------------

    bm25_retriever.k = sparse_k

    sparse_docs = bm25_retriever.invoke(
        query
    )


    # --------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------------------------

    scores = {}

    documents = {}


    # Dense results

    for rank, doc in enumerate(
        dense_docs,
        start=1
    ):

        key = doc.page_content

        scores[key] = scores.get(
            key,
            0
        ) + (1 / (60 + rank))

        documents[key] = doc


    # Sparse results

    for rank, doc in enumerate(
        sparse_docs,
        start=1
    ):

        key = doc.page_content

        scores[key] = scores.get(
            key,
            0
        ) + (1 / (60 + rank))

        documents[key] = doc


    # --------------------------------------------------------
    # SORT RESULTS
    # --------------------------------------------------------

    ranked_documents = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # FINAL DOCUMENTS
    # --------------------------------------------------------

    final_docs = [
        documents[key]
        for key, score in ranked_documents[:final_k]
    ]


    # --------------------------------------------------------
    # RETURN DOCUMENTS + RELEVANCE SCORE
    # --------------------------------------------------------

    return final_docs, max_dense_score


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query):

    topic = detect_topic(query)


    # ========================================================
    # FITNESS
    # ========================================================

    if topic == "fitness":

        docs, relevance_score = hybrid_retrieval(
            query=query,
            vector_db=fitness_db,
            bm25_retriever=fitness_bm25,
            dense_k=3,
            sparse_k=3,
            final_k=4
        )


        return docs, relevance_score


    # ========================================================
    # NUTRITION
    # ========================================================

    elif topic == "nutrition":

        docs, relevance_score = hybrid_retrieval(
            query=query,
            vector_db=nutrition_db,
            bm25_retriever=nutrition_bm25,
            dense_k=3,
            sparse_k=3,
            final_k=4
        )


        return docs, relevance_score


    # ========================================================
    # BOTH
    # ========================================================

    else:

        fitness_docs, fitness_score = hybrid_retrieval(
            query=query,
            vector_db=fitness_db,
            bm25_retriever=fitness_bm25,
            dense_k=2,
            sparse_k=2,
            final_k=2
        )


        nutrition_docs, nutrition_score = hybrid_retrieval(
            query=query,
            vector_db=nutrition_db,
            bm25_retriever=nutrition_bm25,
            dense_k=2,
            sparse_k=2,
            final_k=2
        )


        docs = (
            fitness_docs +
            nutrition_docs
        )


        relevance_score = max(
            fitness_score,
            nutrition_score
        )


        return docs, relevance_score


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, language):


    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    docs, relevance_score = retrieve_documents(
        query
    )


    # --------------------------------------------------------
    # MINIMUM RELEVANCE CHECK
    # --------------------------------------------------------

    RELEVANCE_THRESHOLD = 0.30


    if (
        not docs
        or relevance_score < RELEVANCE_THRESHOLD
    ):

        return (
            "I don't know the answer based on "
            "the provided documents."
        )


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content[:400]
        for doc in docs
    )


    # --------------------------------------------------------
    # STRICT RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a professional fitness and nutrition assistant.

Your ONLY source of information is the provided context.

IMPORTANT RULES:

1. Answer the question ONLY if the answer can be found
   in the provided context.

2. Do NOT use your own general knowledge.

3. Do NOT guess.

4. Do NOT make assumptions.

5. Do NOT invent information.

6. Do NOT add information that is not present
   in the context.

7. Retrieved documents may contain irrelevant information.
   Ignore irrelevant information.

8. If the context does not contain enough information
   to answer the question, respond EXACTLY with:

I don't know the answer based on the provided documents.

9. If only part of the question is supported by the context,
   answer only the supported part.

10. Do not use information from the question itself
    as evidence.

11. Keep the answer clear and concise.

12. Answer in {language}.

Context:
--------------------------------------------------

{context}

--------------------------------------------------

Question:
{query}

Answer:
"""


    # --------------------------------------------------------
    # DEEPSEEK
    # --------------------------------------------------------

    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200,
        temperature=0.1
    )


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    answer = response.choices[0].message.content


    return answer