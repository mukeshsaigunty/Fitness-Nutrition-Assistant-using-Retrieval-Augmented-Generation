import os
import asyncio

from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("hug")

if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in .env")


# ============================================================
# RAGAS
# ============================================================

from ragas.llms import llm_factory

from ragas.embeddings import HuggingFaceEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


# ============================================================
# HUGGING FACE OPENAI-COMPATIBLE CLIENT
#
# IMPORTANT:
# This is Hugging Face.
# We are NOT using OpenAI's API.
# ============================================================

from openai import AsyncOpenAI


hf_client = AsyncOpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
)


# ============================================================
# RAGAS EVALUATION LLM
# ============================================================

print("Loading Ragas evaluation LLM...")

ragas_llm = llm_factory(
    model="deepseek-ai/DeepSeek-V3-0324",
    provider="openai",
    client=hf_client,
    temperature=0.1,
    max_tokens=1000,
)


# ============================================================
# RAGAS EMBEDDINGS
# ============================================================

print("Loading Ragas evaluation embeddings...")

ragas_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE RAGAS METRICS
# ============================================================

faithfulness_metric = Faithfulness(
    llm=ragas_llm
)

answer_relevancy_metric = AnswerRelevancy(
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

context_precision_metric = ContextPrecision(
    llm=ragas_llm
)

context_recall_metric = ContextRecall(
    llm=ragas_llm
)


# ============================================================
# IMPORT RAG SYSTEM
# ============================================================

from rag_system import (
    retrieve_documents,
    generate_answer,
)


# ============================================================
# TEST CASES
# ============================================================

test_cases = [

    {
        "question": "What exercises train shoulders?",

        "reference":
            "The exercises that train shoulders are "
            "Standing Cable Lateral Raise and "
            "Converging Shoulder Press."
    },

]


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

async def evaluate_question(question, reference):

    print("\n")
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    print("\nRetrieving documents...")

    docs, relevance_score = retrieve_documents(question)

    contexts = [
        doc.page_content
        for doc in docs
    ]

    print(
        f"Retrieved contexts: {len(contexts)}"
    )

    print(
        f"Dense relevance score: {relevance_score}"
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        "English"
    )

    print("\n")
    print("=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print(answer)


    # ========================================================
    # FAITHFULNESS
    # ========================================================

    print("\nCalculating Faithfulness...")

    faithfulness_result = await faithfulness_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    faithfulness_score = faithfulness_result.value


    # ========================================================
    # ANSWER RELEVANCY
    # ========================================================

    print("Calculating Answer Relevancy...")

    answer_relevancy_result = await answer_relevancy_metric.ascore(
        user_input=question,
        response=answer,
    )

    answer_relevancy_score = answer_relevancy_result.value


    # ========================================================
    # CONTEXT PRECISION
    # ========================================================

    print("Calculating Context Precision...")

    context_precision_result = await context_precision_metric.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=contexts,
    )

    context_precision_score = context_precision_result.value


    # ========================================================
    # CONTEXT RECALL
    # ========================================================

    print("Calculating Context Recall...")

    context_recall_result = await context_recall_metric.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=contexts,
    )

    context_recall_score = context_recall_result.value


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Faithfulness       : {faithfulness_score:.4f}"
    )

    print(
        f"Answer Relevancy   : {answer_relevancy_score:.4f}"
    )

    print(
        f"Context Precision  : {context_precision_score:.4f}"
    )

    print(
        f"Context Recall     : {context_recall_score:.4f}"
    )

    print("=" * 70)


    return {
        "question": question,
        "answer": answer,
        "faithfulness": faithfulness_score,
        "answer_relevancy": answer_relevancy_score,
        "context_precision": context_precision_score,
        "context_recall": context_recall_score,
    }


# ============================================================
# MAIN
# ============================================================

async def main():

    print("\n")
    print("=" * 70)
    print("RAGAS RAG EVALUATION")
    print("=" * 70)

    results = []


    for test_case in test_cases:

        result = await evaluate_question(
            test_case["question"],
            test_case["reference"],
        )

        results.append(result)


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)


    for result in results:

        print("\nQuestion:")
        print(result["question"])

        print(
            f"Faithfulness      : "
            f"{result['faithfulness']:.4f}"
        )

        print(
            f"Answer Relevancy  : "
            f"{result['answer_relevancy']:.4f}"
        )

        print(
            f"Context Precision : "
            f"{result['context_precision']:.4f}"
        )

        print(
            f"Context Recall    : "
            f"{result['context_recall']:.4f}"
        )


    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())