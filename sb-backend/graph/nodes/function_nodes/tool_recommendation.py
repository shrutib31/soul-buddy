import os
import json
import logging
import psycopg
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Model Flags — flip these to control which model(s) run
# ---------------------------------------------------------------------------
FLAG_COMPARISON_MODE: bool = False   # True → both models run, judge picks winner
FLAG_USE_LLAMA: bool = False        # True → use local Ollama/Llama
FLAG_USE_OPENAI: bool = True        # True → use OpenAI gpt-4o-mini

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

LLAMA_API_URL = os.getenv("OLLAMA_BASE_URL")
DB_CONN = os.getenv("PGVECTOR_CONNECTION_STRING")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Validation — fail fast if flags are misconfigured
# ---------------------------------------------------------------------------
def _validate_flags() -> None:
    if FLAG_COMPARISON_MODE:
        if not (FLAG_USE_LLAMA and FLAG_USE_OPENAI):
            raise ValueError(
                "FLAG_COMPARISON_MODE=True requires both FLAG_USE_LLAMA and "
                "FLAG_USE_OPENAI to be True."
            )
    else:
        if FLAG_USE_LLAMA == FLAG_USE_OPENAI:
            raise ValueError(
                "When FLAG_COMPARISON_MODE=False, exactly one of FLAG_USE_LLAMA "
                "or FLAG_USE_OPENAI must be True."
            )


# ---------------------------------------------------------------------------
# Step 1 — Retrieve relevant context from PGVector
# ---------------------------------------------------------------------------
def retrieve_context(message: str, personality: dict, k: int = 4) -> str:
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text:latest", base_url=LLAMA_API_URL
    )

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
# Shared system prompt builder
# ---------------------------------------------------------------------------
def _build_system_prompt(personality: dict, context: str) -> str:
    return f"""
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


# ---------------------------------------------------------------------------
# Step 2a — Generate recommendation via Llama (Ollama)
# ---------------------------------------------------------------------------
def generate_recommendation_llama(
    personality: dict, context: str, message: str
) -> str:
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url=LLAMA_API_URL,
        temperature=0.3,
    )

    system_prompt = _build_system_prompt(personality, context)

    logger.info("Calling Llama (llama3.2:3b)...")
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ])
        logger.info("Llama response received.")
        return response.content
    except Exception as e:
        logger.error("Llama LLM call failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Step 2b — Generate recommendation via OpenAI (gpt-4o-mini)
# ---------------------------------------------------------------------------
def generate_recommendation_openai(
    personality: dict, context: str, message: str
) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.3,
    )

    system_prompt = _build_system_prompt(personality, context)

    logger.info("Calling OpenAI (gpt-4o-mini)...")
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ])
        logger.info("OpenAI response received.")
        return response.content
    except Exception as e:
        logger.error("OpenAI LLM call failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Step 3 — LLM-as-judge: pick the better response (only in comparison mode)
# ---------------------------------------------------------------------------
def judge_responses(
    llama_response: str,
    openai_response: str,
    message: str,
    personality: dict,
) -> str:
    """
    Sends both responses to gpt-4o-mini acting as a neutral judge.
    Returns whichever response is more helpful, warm, and grounded in context.
    """
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.0,  # deterministic judgment
    )

    judge_prompt = f"""
You are an expert evaluator for a mental wellbeing learning platform.
A user asked the following question:

USER QUESTION: {message}

USER PERSONALITY: {json.dumps(personality, indent=2)}

Two AI assistants produced the following responses:

--- RESPONSE A ---
{llama_response}

--- RESPONSE B ---
{openai_response}

Evaluate both responses strictly on these criteria:
1. Warmth and empathy — does it feel genuinely caring toward this specific user's personality?
2. Accuracy and grounding — does it stick to the actual worksheet content without hallucinating steps?
3. Clarity — is it easy to follow and actionable?
4. Relevance — does it directly address what the user asked?

Return ONLY the full text of the better response with no preamble, no explanation, no labels.
If both are equal, return Response B.
"""

    logger.info("Calling LLM judge (gpt-4o-mini)...")
    try:
        result = judge_llm.invoke([HumanMessage(content=judge_prompt)])
        logger.info("Judge decision received.")
        return result.content
    except Exception as e:
        logger.error("Judge LLM call failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Main pipeline — called by tool_rec.py
# ---------------------------------------------------------------------------
async def run_tool_recommendation(message: str, personality: dict) -> str:
    _validate_flags()

    context = retrieve_context(message=message, personality=personality)

    if FLAG_COMPARISON_MODE:
        # Run both models, let the judge pick the winner
        logger.info("Comparison mode ON — running both models.")
        llama_resp = generate_recommendation_llama(personality, context, message)
        openai_resp = generate_recommendation_openai(personality, context, message)
        return judge_responses(llama_resp, openai_resp, message, personality)

    elif FLAG_USE_OPENAI:
        return generate_recommendation_openai(personality, context, message)

    else:
        return generate_recommendation_llama(personality, context, message)