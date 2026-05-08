import streamlit as st
import os
import json
import numpy as np
from datetime import datetime
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Prior Auth RAG Bot",
    page_icon="🏥",
    layout="wide"
)

# ── Load and index PDFs ───────────────────────────────────
@st.cache_resource
def build_index():
    """
    Loads PDFs, chunks text, builds FAISS index.
    @st.cache_resource means this runs ONCE and stays in memory.
    No need to rebuild every time someone asks a question.
    """
    policies_dir = "data/sample_policies"
    all_chunks = []

    pdf_files = [f for f in os.listdir(policies_dir) if f.endswith(".pdf")]

    for filename in pdf_files:
        filepath = os.path.join(policies_dir, filename)
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            # Split into chunks
            chunk_size = 800
            overlap = 150
            start = 0
            while start < len(text):
                chunk = text[start:start + chunk_size]
                all_chunks.append({
                    "source_file": filename,
                    "page": page_num + 1,
                    "content": chunk
                })
                start += chunk_size - overlap

    # Build TF-IDF vectorizer
    texts = [c["content"] for c in all_chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    vectors = vectorizer.fit_transform(texts).toarray().astype("float32")
    faiss.normalize_L2(vectors)

    # Build FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return all_chunks, vectorizer, index


def search(question, chunks, vectorizer, index, k=3):
    """Finds the top-k most relevant policy chunks for a question."""
    q_vec = vectorizer.transform([question]).toarray().astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "source_file": chunks[idx]["source_file"],
            "page": chunks[idx]["page"],
            "content": chunks[idx]["content"],
            "score": round(float(score), 4)
        })
    return results


def log_query(question, results):
    """Saves query to audit log."""
    os.makedirs("outputs", exist_ok=True)
    log_file = "outputs/query_log.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "sources": [{"file": r["source_file"], "page": r["page"], "score": r["score"]} for r in results]
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main app ──────────────────────────────────────────────
def main():
    # Header
    st.title("🏥 Prior Authorization RAG Bot")
    st.markdown("**Ask any prior authorization eligibility question — answers pulled directly from payer policy documents**")
    st.markdown("---")

    # Load index
    with st.spinner("Loading policy documents..."):
        chunks, vectorizer, index = build_index()

    # Two column layout
    col_main, col_sidebar = st.columns([2, 1])

    with col_main:
        st.subheader("💬 Ask a Question")

        # Sample questions
        st.markdown("**Try one of these:**")
        sample_questions = [
            "Is an MRI of the lumbar spine covered without prior conservative treatment?",
            "What documentation is required for bariatric surgery authorization?",
            "Are telehealth visits covered for behavioral health services?",
            "Does a patient with cauda equina syndrome need 6 weeks of conservative treatment?",
            "What ICD-10 codes are required for knee MRI authorization?"
        ]

        selected = st.selectbox(
            "Select a sample question or type your own below",
            [""] + sample_questions
        )

        question = st.text_input(
            "Or type your question here:",
            value=selected,
            placeholder="e.g. Is an MRI covered without physical therapy?"
        )

        search_btn = st.button("🔍 Search Policy Documents", type="primary")

        if search_btn and question:
            with st.spinner("Searching policy documents..."):
                results = search(question, chunks, vectorizer, index, k=3)
                log_query(question, results)

            st.markdown("---")
            st.subheader("📋 Relevant Policy Sections Found")

            for i, result in enumerate(results, 1):
                relevance_color = "🟢" if result["score"] > 0.2 else "🟡" if result["score"] > 0.1 else "🔴"
                with st.expander(
                    f"{relevance_color} Source {i}: {result['source_file']} — Page {result['page']} (Relevance: {result['score']})",
                    expanded=(i == 1)
                ):
                    st.markdown(result["content"])

            # Show top answer prominently
            st.markdown("---")
            st.subheader("✅ Most Relevant Policy Section")
            st.info(results[0]["content"])
            st.caption(f"Source: {results[0]['source_file']} — Page {results[0]['page']}")

        elif search_btn and not question:
            st.warning("Please enter a question first.")

    with col_sidebar:
        st.subheader("📁 Loaded Policy Documents")
        policies_dir = "data/sample_policies"
        pdf_files = [f for f in os.listdir(policies_dir) if f.endswith(".pdf")]
        for pdf in pdf_files:
            st.markdown(f"📄 {pdf}")

        st.markdown("---")
        st.subheader("📊 Index Stats")
        st.metric("Total Chunks Indexed", len(chunks))
        st.metric("Policy Documents", len(pdf_files))

        st.markdown("---")
        st.subheader("🕐 Recent Queries")
        log_file = "outputs/query_log.jsonl"
        if os.path.exists(log_file):
            with open(log_file) as f:
                logs = [json.loads(line) for line in f.readlines()[-5:]]
            for log in reversed(logs):
                st.caption(f"🔍 {log['question'][:60]}...")
        else:
            st.caption("No queries yet.")

        st.markdown("---")
        st.caption("Built by Neha Pannala | AI Engineer")
        st.caption("github.com/Nehareddy30/prior-auth-rag-bot")


if __name__ == "__main__":
    main()
