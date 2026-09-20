import os

import streamlit as st
from dotenv import load_dotenv

from api_client import APIError, ask_question, check_health

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL")

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="centered")
st.title("📄 Chat with your Documents")
st.caption("Answers are generated only from the indexed documents and cite their sources.")

if not API_BASE_URL:
    st.error("API_BASE_URL is not set. Copy `.env.example` to `.env` and set the backend URL.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Backend")
    st.code(API_BASE_URL)
    if st.button("Check connection"):
        try:
            info = check_health(API_BASE_URL)
            st.success(f"Online - {info.get('chunks', '?')} chunks indexed")
        except APIError as exc:
            st.error(str(exc))
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def render_sources(sources: list[str]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                st.markdown(f"- {source}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

if question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents and generating an answer..."):
                result = ask_question(API_BASE_URL, question)
        except APIError as exc:
            st.error(str(exc))
        else:
            st.markdown(result["answer"])
            render_sources(result["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
