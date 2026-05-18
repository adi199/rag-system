import os
import html
import json
from groq import Groq
import re
import psycopg
from pgvector.psycopg import register_vector
import streamlit as st
import transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from db import DB_KWARGS
import logging
import time

load_dotenv()
transformers.utils.logging.set_verbosity_error()

# Module-level logger setup
logger = logging.getLogger(__name__)

# ==========================================
# 1. Initialization and Setup
# ==========================================

def inject_custom_css():
    st.markdown("""
    <style>
    citation {
        display: inline-block;
        background-color: rgba(255, 75, 75, 0.1);
        color: #ff4b4b;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.8em;
        font-weight: 500;
        margin: 0 4px;
        border: 1px solid rgba(255, 75, 75, 0.2);
        cursor: help;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_llm_client():
    return Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )

@st.cache_resource
def get_embedding_model():
    """Load the same SentenceTransformer model used during ingestion."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================================
# 2. Database Operations
# ==========================================

def query_database(query_text: str, embed_model: SentenceTransformer, cross_encoder, n_results: int = 15, top_k: int = 10) -> list[dict]:
    """Runs hybrid_search RPC on Supabase and reranks with CrossEncoder."""
    start_time = time.time()
    
    # 1. Encode the query with the same model used during ingestion
    query_embedding = embed_model.encode(
        query_text, normalize_embeddings=True
    ).tolist()

    # 2. Call the hybrid_search Postgres function
    # IMPORTANT: autocommit=True prevents psycopg from issuing an implicit BEGIN,
    # which would leave the connection "idle in transaction" during the CPU-heavy
    # CrossEncoder.predict() call below, pinning the PgBouncer backend connection.
    db_start = time.time()
    rows = []
    with psycopg.connect(**DB_KWARGS, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, content, metadata
                FROM hybrid_search(
                    %s,
                    %s::vector,
                    %s
                )
                """,
                (query_text, query_embedding, n_results),
            )
            rows = cur.fetchall()
    db_duration = time.time() - db_start

    docs = []
    for chunk_id, content, metadata in rows:
        docs.append({
            "id": chunk_id,
            "content": content,
            "metadata": metadata if isinstance(metadata, dict) else json.loads(metadata),
        })

    if not docs:
        logger.info(f"Database search returned 0 results for: '{query_text}' (DB search took {db_duration:.3f}s)")
        return []

    # 3. Rerank with CrossEncoder — runs AFTER the connection is fully closed
    rerank_start = time.time()
    pairs = [[query_text, doc['content']] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    rerank_duration = time.time() - rerank_start
    
    total_duration = time.time() - start_time
    logger.info(
        f"Query DB & Rerank completed in {total_duration:.3f}s for: '{query_text}' "
        f"[DB: {db_duration:.3f}s, Rerank: {rerank_duration:.3f}s, "
        f"Input chunks: {len(docs)}, Selected top_k: {min(top_k, len(docs))}]"
    )
    
    return [doc for doc, score in scored_docs[:top_k]]

# ==========================================
# 3. LLM Interaction & Query Decomposition
# ==========================================

def decompose_query(llm_client: Groq, query: str) -> list[str]:
    """Uses the LLM to decompose a complex query into simpler sub-queries for better retrieval."""
    start_time = time.time()
    system_prompt = """You are an expert AI search assistant. Your task is to decompose complex, multi-part questions into 2–3 focused, keyword-dense search queries that can each be answered independently.
    When decomposing, consider:
    - Whether the question spans multiple topics, time periods, or company versions that should be searched separately
    - Whether conflicting or comparative information might exist across different documents or versions
    - If the question is already simple and single-topic, extract the core keywords.
    - CRITICAL: Generate search queries as optimized keywords or factual statements, NOT as natural language questions. (e.g., "Digital Realty customers December 2025" instead of "How many customers did Digital Realty have in December 2025?")
    - Output ONLY the sub-queries, one per line. No bullet points, numbers, labels, or introductory text."""
    
    response = llm_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    
    # Split the output by lines and filter out empty strings
    content = response.choices[0].message.content or ""
    sub_queries = [q.strip("- *1234567890.") for q in content.split('\n') if q.strip()]
    
    duration = time.time() - start_time
    logger.info(f"Query decomposed in {duration:.3f}s. Input: '{query}' -> Sub-queries: {sub_queries}")
    
    return sub_queries if sub_queries else [query]

def build_prompt(query: str, context_docs: list[dict], chat_history: list[dict]) -> str:
    """Constructs the prompt with context for the LLM."""
    context_parts = []
    for doc in context_docs:
        chunk_id = doc["id"]
        content = html.unescape(doc['content'])
        # Completely strip HTML tags so the LLM doesn't even see them
        content = re.sub(r'<[^>]+>', ' | ', content)
        context_parts.append(f"Source [{chunk_id}]:\n{content}")
        
    context_str = "\n\n---\n\n".join(context_parts)
    
    history_str = ""
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent_history:
        history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    return f"""You are a precise and trustworthy AI research assistant. Answer the user's question using ONLY the provided context documents.

            SOURCING & CITATION
            - For each metric, fact, number, or statistic, it is MANDATORY to provide its source immediately after in this exact format: [Source: chunk_id].
            - EXCEPTION: If a metric has been calculated by you based on the context, a citation is not required for that specific calculated metric.
            - If citing multiple sources, you MUST use separate brackets for each source. DO NOT comma-separate them.
            - Never fabricate citations. Only use chunk IDs present in the context documents.
            - Always attribute claims to their specific source document.

            VERSION AWARENESS & CONFLICTS
            - Context documents may represent different versions or time periods of the same company or subject.
            - Never silently blend values from different versions. Report each separately with its source.
            - Explicitly note which version or time period a piece of information belongs to when relevant.
            - If documents contain contradictory information, surface the conflict explicitly.

            GAPS & PARTIAL ANSWERS
            - If the context only partially addresses the question, answer what you can and clearly flag the gaps.
            - Never invent or infer information not present in the context.
            - If a figure can be computed or inferred from numbers present in the context, do so and show your reasoning. Refusal is only appropriate when the data is entirely absent.

            Do not over-explain or over-elaborate. Be concise and to the point.
            
            ---
            Context Documents:
            {context_str}

            ---
            Recent Conversation History:
            {history_str}

            ---
            User Question:
            {query}

            ---
            FINAL FORMATTING INSTRUCTIONS (STRICT)
            You must format your final response following these exact rules:
            1. Use ONLY sub-headings, paragraphs, bullet points, and tables. 
            2. Do NOT copy raw image tags (like `![...]`), raw HTML tags, or complex nested markdown from the context documents. Keep the structure clean and readable.
            3. Every metric or statistic must have its citation appended.
            4. TABLES: You MUST use standard Markdown table syntax (using `|` and `-`). 
               - DO NOT output any HTML table tags (`<table>`, `<tr>`, `<td>`, etc.).
               - DO NOT output encoded HTML entities (like `&lt;` or `&gt;`).
               - Convert any tabular data from the context into a clean Markdown table.

            Example of Good Formatting:
            ### Financial Overview
            The total revenue was $10M [Source: doc_A-chunk1], representing an increase from previous years. 

            ### Key Metrics
            - Gross Margin: 45% [Source: doc_B-chunk2]
            - Operating Costs: $2M [Source: doc_B-chunk2]

            Now, generate your answer following these rules exactly."""

def generate_response_stream(llm_client: Groq, prompt: str):
    """Calls the Groq API to generate an answer based on the prompt and returns the stream."""
    return llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt}
        ],
        stream=True,
        temperature=0
    )

def process_citations(text: str, context_docs: list[dict]) -> str:
    """Replaces [Source: chunk_id] with interactive <citation> tags."""
    source_map = {}
    for doc in context_docs:
        chunk_id = doc["id"]
        meta = doc.get("metadata", {})
        source_name = meta.get("file_name", meta.get("source", "Document"))
        page = meta.get("page_number", meta.get("page", ""))
        source_ref = f"{source_name} (Page {page})" if page and page != "N/A" else source_name
        
        # Create a brief preview of the content for the title tooltip
        content_preview = html.unescape(doc['content'])
        content_preview = re.sub(r'<[^>]+>', ' ', content_preview)[:250]
        content_preview = content_preview.replace('"', '&quot;').replace('\n', ' ') + "..."
        
        source_map[chunk_id] = {
            "ref": source_ref,
            "title": content_preview
        }
        
    def replacer(match):
        # Support comma-separated multiple chunk IDs
        chunk_ids = [c.strip() for c in match.group(1).split(',')]
        citations = []
        for cid in chunk_ids:
            if cid in source_map:
                ref = source_map[cid]["ref"]
                title = source_map[cid]["title"]
                citations.append(f'<citation title="{title}">{ref}</citation>')
            else:
                citations.append(f'<citation>{cid}</citation>')
        return " " + " ".join(citations)
        
    pattern = r'\[Source:\s*(.+?)\]'
    return re.sub(pattern, replacer, text)

# ==========================================
# 4. Streamlit UI
# ==========================================

def main():
    st.set_page_config(page_title="RAG Chat Assistant", page_icon="🤖")
    st.title("📚 RAG Chat Assistant")
    st.caption("Chat with your documents grounded by Supabase and Groq.")
    
    inject_custom_css()
    
    # Initialize clients
    try:
        llm_client = get_llm_client()
        embed_model = get_embedding_model()
        cross_encoder = get_cross_encoder()
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "rendered_content" in message:
                st.markdown(message["rendered_content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"], unsafe_allow_html=True)
            
    # Handle user input
    if user_query := st.chat_input("Ask a question about your documents..."):
        # 1. Add user message to history and UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # 2. Process Assistant response
        with st.chat_message("assistant"):
            logger.info(f"Processing user query: '{user_query}'")
            
            with st.status("Analyzing query and searching documents...", expanded=True) as status:
                st.write("Decomposing complex query into simple key-phrase searches...")
                sub_queries = decompose_query(llm_client, user_query)
                st.write(f"Generated sub-queries: {', '.join([f'`{q}`' for q in sub_queries])}")
                
                # Fetch and deduplicate chunks
                all_context_docs = []
                seen_contents = set()
                
                # Use an empty placeholder to update search message in-place (no layout clutter)
                search_status_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                
                for idx, sub_q in enumerate(sub_queries):
                    search_status_placeholder.write(f"Searching database & reranking for: `{sub_q}`...")
                    try:
                        # Each sub-query gets its own short-lived autocommit connection.
                        # This ensures the backend connection is released back to PgBouncer
                        # immediately after fetchall(), before the blocking CrossEncoder call.
                        docs = query_database(sub_q, embed_model, cross_encoder, n_results=20, top_k=5)
                    except Exception as db_err:
                        logger.exception(f"Database error for sub-query '{sub_q}'")
                        st.error(f"Database error: {db_err}")
                        docs = []
                    for d in docs:
                        if d['content'] not in seen_contents:
                            seen_contents.add(d['content'])
                            all_context_docs.append(d)
                    
                    progress_bar.progress((idx + 1) / len(sub_queries))
                
                progress_bar.empty()
                search_status_placeholder.empty()
                
                context_docs = all_context_docs
                status.update(label=f"Retrieved {len(context_docs)} unique context chunks!", state="complete", expanded=False)
                
            if not context_docs:
                st.markdown("I couldn't find any relevant information in the documents to answer your question.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "I couldn't find any relevant information in the documents to answer your question."
                })
                logger.warning(f"No relevant context chunks found for query: '{user_query}'")
            else:
                logger.info(f"Retrieved {len(context_docs)} chunks. Proceeding to LLM response generation.")
                with st.spinner("Thinking..."):
                    prompt = build_prompt(user_query, context_docs, st.session_state.messages[:-1])
                
                # Stream generation
                message_placeholder = st.empty()
                gen_start_time = time.time()
                response_stream = generate_response_stream(llm_client, prompt)
                
                full_response = ""
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                
                gen_duration = time.time() - gen_start_time
                logger.info(
                    f"LLM streaming finished in {gen_duration:.3f}s. "
                    f"Generated {len(full_response)} chars ({len(full_response.split())} words)."
                )
                
                # Replace text citations with interactive HTML pills
                formatted_response = process_citations(full_response, context_docs)
                message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "rendered_content": formatted_response
                })

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence verbose dependency logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    
    main()
