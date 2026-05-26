"""
Streamlit chat UI for the Premier League RAG assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag import answer, load_index

# ---- Page setup ------------------------------------------------------------
st.set_page_config(
    page_title="Premier League RAG Assistant",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Premier League Handbook Assistant")
st.caption(
    "Ask questions about the Premier League and get answers grounded in the "
    "official documents, with citations."
)

# ---- Load the FAISS index once and cache the resource ----------------------
@st.cache_resource
def get_index():
    return load_index()


try:
    index, chunks = get_index()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Open a terminal in this folder and run: `python ingest.py`")
    st.stop()

# ---- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.subheader("Index")
    st.success(f"{len(chunks):,} chunks loaded")

    sources_seen = sorted({src for _, src, _ in chunks})
    st.write("**Documents:**")
    for src in sources_seen:
        st.write(f"- {src}")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Built with Gemini embeddings (gemini-embedding-001) + FAISS + "
        "Gemini 2.5 Flash-Lite."
    )

# ---- Chat state ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src, loc, score in msg["sources"]:
                    st.write(f"- **{src}** {loc}  (similarity: {score:.3f})")

# Chat input
if prompt := st.chat_input("Ask about the Premier League..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                result = answer(prompt, index, chunks)
                st.markdown(result["answer"])
                with st.expander(f"Sources ({len(result['sources'])})"):
                    for src, loc, score in result["sources"]:
                        st.write(f"- **{src}** {loc}  (similarity: {score:.3f})")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except Exception as e:
                error_msg = (
                    f"Sorry, something went wrong: `{type(e).__name__}`. "
                    "The Gemini free tier sometimes hits demand spikes — "
                    "please try again in a moment."
                )
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )