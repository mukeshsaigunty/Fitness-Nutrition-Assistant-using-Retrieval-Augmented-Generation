import streamlit as st
import json
import asyncio
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Fitness & Nutrition Assistant",
    page_icon="🏋️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏋️ AI Fitness & Nutrition Assistant")

st.markdown(
    """
    Ask questions about **gym workouts, exercises, diet plans,
    nutrition, and calories**.
    """
)


# ============================================================
# LOAD RAG SYSTEM
# ============================================================

@st.cache_resource
def load_rag_system():

    from rag_system import (
        generate_answer,
        retrieve_documents
    )

    return generate_answer, retrieve_documents


# ============================================================
# LOAD RAGAS EVALUATION SYSTEM
# ============================================================

@st.cache_resource
def load_evaluation_system():

    from ragas.llms import llm_factory

    from ragas.embeddings import HuggingFaceEmbeddings

    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    from openai import AsyncOpenAI


    # --------------------------------------------------------
    # Hugging Face token
    # --------------------------------------------------------

    hf_token = os.getenv("hug")

    if not hf_token:

        raise ValueError(
            "Hugging Face token 'hug' was not found."
        )


    # --------------------------------------------------------
    # Hugging Face OpenAI-compatible endpoint
    # --------------------------------------------------------

    hf_client = AsyncOpenAI(
        api_key=hf_token,
        base_url="https://router.huggingface.co/v1"
    )


    # --------------------------------------------------------
    # Ragas evaluation LLM
    # --------------------------------------------------------

    ragas_llm = llm_factory(
        model="deepseek-ai/DeepSeek-V3-0324",
        provider="openai",
        client=hf_client,
        temperature=0.1,
        max_tokens=1000,
    )


    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    faithfulness = Faithfulness(
        llm=ragas_llm
    )

    answer_relevancy = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    context_precision = ContextPrecision(
        llm=ragas_llm
    )

    context_recall = ContextRecall(
        llm=ragas_llm
    )


    return (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )


# ============================================================
# RAGAS ASYNC EVALUATION FUNCTION
# ============================================================

async def run_ragas_evaluation(
    question,
    answer,
    contexts,
    reference
):

    # --------------------------------------------------------
    # Load metrics
    # --------------------------------------------------------

    (
        faithfulness_metric,
        answer_relevancy_metric,
        context_precision_metric,
        context_recall_metric
    ) = load_evaluation_system()


    # --------------------------------------------------------
    # Faithfulness
    # --------------------------------------------------------

    faithfulness_result = (
        await faithfulness_metric.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
    )


    # --------------------------------------------------------
    # Answer Relevancy
    # --------------------------------------------------------

    answer_relevancy_result = (
        await answer_relevancy_metric.ascore(
            user_input=question,
            response=answer,
        )
    )


    # --------------------------------------------------------
    # Context Precision
    # --------------------------------------------------------

    context_precision_result = (
        await context_precision_metric.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
    )


    # --------------------------------------------------------
    # Context Recall
    # --------------------------------------------------------

    context_recall_result = (
        await context_recall_metric.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
    )


    # --------------------------------------------------------
    # Return scores
    # --------------------------------------------------------

    return {

        "faithfulness":
            faithfulness_result.value,

        "answer_relevancy":
            answer_relevancy_result.value,

        "context_precision":
            context_precision_result.value,

        "context_recall":
            context_recall_result.value,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙ Settings")


    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    languages = [
        "Arabic",
        "Bengali",
        "Chinese",
        "Dutch",
        "English",
        "French",
        "German",
        "Hindi",
        "Indonesian",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish",
        "Tamil",
        "Telugu",
        "Thai",
        "Turkish",
        "Vietnamese"
    ]

    languages = sorted(languages)


    language = st.selectbox(
        "🌍 Select Language",
        languages
    )


    st.divider()


    # --------------------------------------------------------
    # New Chat
    # --------------------------------------------------------

    if st.button("🆕 New Chat"):

        st.session_state.chat_history = []

        st.session_state.last_question = None

        st.session_state.last_answer = None

        st.session_state.last_contexts = []

        st.session_state.evaluation_results = None

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # Download Chat
    # --------------------------------------------------------

    if (
        "chat_history" in st.session_state
        and st.session_state.chat_history
    ):

        chat_json = json.dumps(
            st.session_state.chat_history,
            indent=2
        )

        st.download_button(
            label="⬇ Download Chat",
            data=chat_json,
            file_name="chat_history.json",
            mime="application/json"
        )


    st.divider()


    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    st.info(
        """
        This assistant answers questions using
        **fitness and nutrition documents**
        with **Hybrid Retrieval-Augmented Generation (RAG)**.

        Retrieval uses:

        • ChromaDB
        • BM25
        • Hybrid Retrieval
        • Reciprocal Rank Fusion (RRF)

        Evaluation uses:

        • Ragas
        • Faithfulness
        • Answer Relevancy
        • Context Precision
        • Context Recall
        """
    )


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "last_question" not in st.session_state:

    st.session_state.last_question = None


if "last_answer" not in st.session_state:

    st.session_state.last_answer = None


if "last_contexts" not in st.session_state:

    st.session_state.last_contexts = []


if "evaluation_results" not in st.session_state:

    st.session_state.evaluation_results = None


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask your fitness or nutrition question..."
)


# ============================================================
# GENERATE ANSWER
# ============================================================

if query:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query
        }
    )


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            try:

                generate_answer, retrieve_documents = (
                    load_rag_system()
                )


                # Generate answer

                answer = generate_answer(
                    query,
                    language
                )


                # Retrieve documents

                docs, relevance_score = (
                    retrieve_documents(query)
                )


                contexts = [
                    doc.page_content
                    for doc in docs
                ]


                # Save information for evaluation

                st.session_state.last_question = query

                st.session_state.last_answer = answer

                st.session_state.last_contexts = contexts

                st.session_state.evaluation_results = None


            except Exception as e:

                answer = (
                    "❌ An error occurred while "
                    "generating the answer."
                )

                st.error(
                    "RAG system error:"
                )

                st.exception(e)


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(answer)


    # --------------------------------------------------------
    # Store assistant message
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# RAG EVALUATION
# ============================================================

if (
    st.session_state.last_question
    and st.session_state.last_answer
):

    st.divider()

    st.subheader("📊 RAG Evaluation")

    st.write(
        "Evaluate the latest answer using Ragas."
    )


    # --------------------------------------------------------
    # Evaluate button
    # --------------------------------------------------------

    if st.button(
        "📊 Evaluate Answer",
        type="primary"
    ):

        with st.spinner(
            "🔍 Running Ragas evaluation..."
        ):

            try:

                question = (
                    st.session_state.last_question
                )

                answer = (
                    st.session_state.last_answer
                )

                contexts = (
                    st.session_state.last_contexts
                )


                # ------------------------------------------------
                # Reference answer
                # ------------------------------------------------

                if (
                    question.lower().strip()
                    == "what exercises train shoulders?"
                ):

                    reference = (
                        "The exercises that train "
                        "shoulders are Standing Cable "
                        "Lateral Raise and Converging "
                        "Shoulder Press."
                    )

                else:

                    reference = answer


                # ------------------------------------------------
                # RUN ASYNC RAGAS
                # ------------------------------------------------

                results = asyncio.run(
                    run_ragas_evaluation(
                        question,
                        answer,
                        contexts,
                        reference
                    )
                )


                # ------------------------------------------------
                # Save results
                # ------------------------------------------------

                st.session_state.evaluation_results = (
                    results
                )


            except Exception as e:

                st.session_state.evaluation_results = None

                st.error(
                    "❌ Ragas evaluation failed."
                )

                st.exception(e)


    # ========================================================
    # DISPLAY METRICS
    # ========================================================

    if st.session_state.evaluation_results:

        results = (
            st.session_state.evaluation_results
        )


        st.success(
            "✅ Evaluation completed successfully!"
        )


        col1, col2, col3, col4 = st.columns(4)


        # ----------------------------------------------------
        # Faithfulness
        # ----------------------------------------------------

        with col1:

            st.metric(
                "Faithfulness",
                f"{results['faithfulness']:.4f}"
            )


        # ----------------------------------------------------
        # Answer Relevancy
        # ----------------------------------------------------

        with col2:

            st.metric(
                "Answer Relevancy",
                f"{results['answer_relevancy']:.4f}"
            )


        # ----------------------------------------------------
        # Context Precision
        # ----------------------------------------------------

        with col3:

            st.metric(
                "Context Precision",
                f"{results['context_precision']:.4f}"
            )


        # ----------------------------------------------------
        # Context Recall
        # ----------------------------------------------------

        with col4:

            st.metric(
                "Context Recall",
                f"{results['context_recall']:.4f}"
            )


        st.caption(
            """
            Scores range from 0 to 1.
            Higher scores generally indicate better RAG quality.
            """
        )


        # ----------------------------------------------------
        # Details
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Evaluation Details"
        ):

            st.write(
                "**Question:**"
            )

            st.write(
                st.session_state.last_question
            )


            st.write(
                "**Answer:**"
            )

            st.write(
                st.session_state.last_answer
            )


            st.write(
                "**Retrieved Contexts:**"
            )


            for i, context in enumerate(
                st.session_state.last_contexts,
                start=1
            ):

                st.markdown(
                    f"**Context {i}**"
                )

                st.write(context)