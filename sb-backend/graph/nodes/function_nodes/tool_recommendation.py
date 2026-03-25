import os
import json
import logging
import psycopg
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

LLAMA_3B_API_URL = os.getenv("OLLAMA_BASE_URL")
DB_CONN = os.getenv("PGVECTOR_CONNECTION_STRING")


# ---------------------------------------------------------------------------
# Step 1 — Retrieve relevant context from PGVector
# ---------------------------------------------------------------------------
def retrieve_context(message: str, personality: dict, k: int = 4) -> str:
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=LLAMA_3B_API_URL)

    enriched_query = f"User personality: {json.dumps(personality)}. Question: {message}"
    query_vector = embeddings_model.embed_query(enriched_query)
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    sql = """
        SELECT text
        FROM n8n_vectors
        WHERE metadata->>'chunk_type' = 'worksheet'
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    try:
        with psycopg.connect(DB_CONN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_literal, k))
                rows = cur.fetchall()
        logger.info("Retrieval done — %d chunks fetched.", len(rows))
        return "\n\n".join([row[0] for row in rows])
    except Exception as e:
        logger.error("DB retrieval failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Step 2 — Generate recommendation via LLM
# ---------------------------------------------------------------------------
def generate_recommendation(personality: dict, context: str, message: str) -> str:
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url=LLAMA_3B_API_URL,
        temperature=0.3,
    )

    system_prompt = f"""
You are a warm, caring learning companion speaking directly to the user.
Your ONLY source of knowledge is the course worksheet excerpts below — do not use anything outside them.

--- USER PERSONALITY PROFILE ---
{json.dumps(personality, indent=2)}

--- COURSE WORKSHEET EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Respond as one flowing paragraph (no headings, no bullet points, no quotes):
1. Start with 1–2 warm sentences validating what the user is feeling or asking, tailored to their personality.
2. Then name the specific worksheet or tool from the excerpts and explain in 1–2 sentences why it suits this user.
3. Then walk them through the key steps from that worksheet in plain, encouraging language.

Rules:
- Speak directly to the user as "you".
- Only use content from the excerpts above — never invent steps or tools.
- Keep the entire response under 200 words.
- If the excerpts have no relevant content, say so warmly.
"""

    logger.info("Calling LLM (llama3.2:3b)...")
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ])
        logger.info("LLM response received.")
        return response.content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Main pipeline — called by tool_rec.py
# ---------------------------------------------------------------------------
async def run_tool_recommendation(message: str, personality: dict) -> str:
    context = retrieve_context(message=message, personality=personality)
    return generate_recommendation(personality=personality, context=context, message=message)
