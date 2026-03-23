import os
import json
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
# --- PGVector connection string (Postgres DB) --------------------
PGVECTOR_CONNECTION_STRING =  os.getenv("PGVECTOR_CONNECTION_STRING")
# --- Collection name (the table your embeddings are stored in) -------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

LLAMA_3B_API_URL= os.getenv("OLLAMA_BASE_URL")
LLAMA_MODEL_TAG="llama3.2:3b"

# ===========================================================================


def fix_connection_string(url: str) -> str:
    """
    LangChain PGVector needs postgresql+psycopg:// format.
    This converts postgres:// or postgresql:// automatically.
    """
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# ---------------------------------------------------------------------------
# Step 1 — Retrieve relevant context from PGVector
# ---------------------------------------------------------------------------
def retrieve_context(message: str, personality: dict, k: int = 4) -> str:
    """
    Embeds the user message using Ollama nomic-embed-text,
    does similarity search against your pgvector store,
    and returns the top-k relevant chunks as a single string.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url = LLAMA_3B_API_URL)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="n8n_vectors",
        connection=fix_connection_string("postgres://postgres:postgres123@72.60.99.35:5433/lifeskills_rag"),
    )

    # Build a richer query using personality + message for better retrieval
    personality_summary = json.dumps(personality)
    enriched_query = f"User personality: {personality_summary}. Question: {message}"

    docs = vector_store.similarity_search(enriched_query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])
    return context


# ---------------------------------------------------------------------------
# Step 2 — Generate recommendation via LLM
# ---------------------------------------------------------------------------
def generate_recommendation(
    personality: dict,
    context: str,
    message: str,
) -> str:
    """
    Passes personality + retrieved context + user message to the LLM
    and returns a personalized tool recommendation.
    """
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url=LLAMA_3B_API_URL,
        temperature=0.7,
    )

    system_prompt = f"""
You are a personalized learning assistant. Your job is to recommend the most 
suitable tools and courses to a user based on their personality profile and 
the available course knowledge base.

--- USER PERSONALITY PROFILE ---
{json.dumps(personality, indent=2)}

--- RELEVANT COURSE KNOWLEDGE BASE CONTEXT ---
{context}

Using the personality profile and the context above, recommend the most 
relevant tools or courses for the user. Be specific, helpful, and concise.
Tailor your recommendations to match the user's personality traits and goals.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message),
    ]

    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Main pipeline — called by tool_rec.py
# ---------------------------------------------------------------------------
async def run_tool_recommendation(
    message: str,
    personality: dict,
) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant context from PGVector using message + personality
    2. Generate personalized tool recommendation via LLM
    """

    # Step 1 — retrieve context from vector store
    context = retrieve_context(message=message, personality=personality)

    # Step 2 — generate recommendation
    recommendation = generate_recommendation(
        personality=personality,
        context=context,
        message=message,
    )

    return recommendation