import hmac
import html
import itertools
import logging
import os
import re

import streamlit as st
from dotenv import load_dotenv

from api_client import (
    UNAVAILABLE_MESSAGE,
    APIError,
    delete_document,
    get_health,
    list_documents,
    stream_query,
    upload_document,
)

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL")
# Technical details (backend URL, model, chunk counts, similarity, latency) are for developers only.
DEBUG = os.getenv("DEBUG_UI", "").strip().lower() in {"1", "true", "yes", "on"}
# Optional shared password for the whole app (recommended when the app is public).
APP_PASSWORD = os.getenv("APP_PASSWORD") or None
# Read-only mode for public demos: visitors can browse the documents but not add or remove them.
READ_ONLY = os.getenv("READ_ONLY_DOCUMENTS", "").strip().lower() in {"1", "true", "yes", "on"}
APP_NAME = "DocuMind"
DEFAULT_EXAMPLES_EN = [
    "What is the F1 score?",
    "How does the elbow method help choose k?",
    "What is the difference between L1 and L2 regularization?",
    "What is data leakage and how can I avoid it?",
]
DEFAULT_EXAMPLES_AR = [
    "ما هو مقياس F1؟",
    "ما الفرق بين التصنيف (Classification) والانحدار (Regression)؟",
    "كيف تساعد طريقة الكوع في اختيار عدد العناقيد في k-means؟",
    "ما الفرق بين التنظيم L1 وL2؟",
    "ما هو تسرّب البيانات وكيف أتجنبه؟",
    "لماذا تكون الدقة مضللة مع البيانات غير المتوازنة؟",
]


def _examples(env_name: str, default: list[str]) -> list[str]:
    return [q.strip() for q in os.getenv(env_name, "").split("|") if q.strip()] or default


EXAMPLES_EN = _examples("EXAMPLE_QUESTIONS", DEFAULT_EXAMPLES_EN)
EXAMPLES_AR = _examples("EXAMPLE_QUESTIONS_AR", DEFAULT_EXAMPLES_AR)
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

st.set_page_config(
    page_title=f"{APP_NAME} | Document Assistant",
    page_icon="📄",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": f"**{APP_NAME}**: ask questions about your documents and get answers with cited sources.",
    },
)

CSS = """
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden; display: none;}
.block-container, [data-testid="stMainBlockContainer"] {max-width: 1000px; padding-top: 2rem;}
.hero {background: linear-gradient(135deg, #0B3C49 0%, #1C6E7D 100%); border-radius: 16px;
       padding: 26px 32px; margin-bottom: 1.2rem;}
.hero h1 {color: #FFFFFF; margin: 0; padding: 0; font-size: 1.8rem; line-height: 1.25;}
.hero p {color: #CADCE0; margin: 8px 0 14px 0; font-size: 1rem;}
.hero .tag {display: inline-block; background: rgba(242,165,65,.18); color: #F2A541; border-radius: 999px;
            padding: 3px 12px; margin-right: 8px; font-size: .78rem; font-weight: 600;}
.brand {font-size: 1.35rem; font-weight: 700; color: #0B3C49;}
.brand-sub {color: #5B7480; font-size: .85rem; margin-bottom: .6rem;}
.pill {display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: .8rem; font-weight: 600;}
.pill.ok {background: rgba(46,160,67,.14); color: #1a7f37;}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
        background: rgba(28,110,125,.12); color: #1C6E7D; margin: 4px 6px 4px 0;}
.src-card {border: 1px solid rgba(128,128,128,.28); border-left: 4px solid #F2A541; border-radius: 10px;
           padding: 10px 14px; margin: 8px 0; background: rgba(28,110,125,.06);}
.src-title {font-weight: 600; font-size: .92rem;}
.src-title .ref {display: inline-block; background: #F2A541; color: #0B3C49; border-radius: 6px;
                 padding: 0 7px; margin-right: 8px; font-size: .8rem;}
.src-meta {color: #5B7480; font-size: .78rem; margin-top: 2px;}
.src-snippet {font-size: .85rem; margin-top: 6px; opacity: .92;}
.msg-rtl {font-family: "Segoe UI", Tahoma, "Noto Naskh Arabic", "Noto Sans Arabic", Arial, sans-serif;
         line-height: 1.9; font-size: 1.03rem;}
.hero .ar {color: #CADCE0; margin: 0 0 14px 0; font-size: 1rem; direction: rtl; text-align: right;}
[data-testid="stChatInput"] textarea {unicode-bidi: plaintext;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if not API_BASE_URL:
    # A deployment problem, not something a user can fix: log it, show a plain message.
    logging.getLogger(__name__).error("API_BASE_URL is not set. Copy frontend/.env.example to .env.")
    st.error("The assistant isn't available right now. Please contact the administrator.")
    if DEBUG:
        st.caption("API_BASE_URL is not set. Copy `.env.example` to `.env`.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)


def require_login() -> None:
    """Show a sign-in screen until the shared APP_PASSWORD is entered (no-op when it is not set)."""
    if not APP_PASSWORD or st.session_state.get("authenticated"):
        return
    st.markdown(
        '<div class="hero"><h1>Sign in to DocuMind</h1>'
        '<p>Enter the password to continue.</p>'
        '<p class="ar">أدخل كلمة المرور للمتابعة.</p></div>',
        unsafe_allow_html=True,
    )
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(password.encode("utf-8"), APP_PASSWORD.encode("utf-8")):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_login()


# ----------------------------------------------------------------- helpers
def esc(text: str) -> str:
    return html.escape(str(text)).replace("$", "&#36;")


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def rtl_html(text: str) -> str:
    """Render text (with light **bold** and bullet support) as an auto-direction block."""
    body = esc(text)
    body = re.sub(r"(?m)^[ \t]*[-*][ \t]+", "&bull; ", body)
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    return f'<div class="msg-rtl" dir="auto">{body.replace(chr(10), "<br>")}</div>'


def render_text(text: str) -> None:
    if has_arabic(text):
        st.markdown(rtl_html(text), unsafe_allow_html=True)
    else:
        st.markdown(text)


def stream_rtl(tokens) -> str:
    """Stream an Arabic answer into a right-to-left block (st.write_stream is left-to-right only)."""
    box, buffer = st.empty(), ""
    for token in tokens:
        buffer += token
        box.markdown(rtl_html(buffer + " ▌"), unsafe_allow_html=True)
    box.markdown(rtl_html(buffer), unsafe_allow_html=True)
    return buffer


def show_error(exc: APIError) -> None:
    st.error(str(exc))
    if DEBUG and getattr(exc, "technical", None):
        st.caption(exc.technical)


def source_cards_html(citations: list[dict]) -> str:
    cards = []
    for c in citations:
        detail = f"Page {c['page']}"
        if DEBUG:
            detail += f" &middot; similarity {c['score']:.2f}"
        cards.append(
            '<div class="src-card">'
            f'<div class="src-title" dir="auto"><span class="ref">{c["ref"]}</span>{esc(c["source"])}</div>'
            f'<div class="src-meta">{detail}</div>'
            f'<div class="src-snippet" dir="auto">{esc(c["snippet"])}</div></div>'
        )
    return "".join(cards)


def render_extras(message: dict) -> None:
    citations = message.get("citations") or []
    badges = []
    if citations:
        badges.append(f'<span class="badge">{len(citations)} source{"s" if len(citations) != 1 else ""}</span>')
    if DEBUG:
        if message.get("latency_ms") is not None:
            badges.append(f'<span class="badge">{message["latency_ms"] / 1000:.1f} s</span>')
        if message.get("model"):
            badges.append(f'<span class="badge">{esc(message["model"])}</span>')
    if badges:
        st.markdown("".join(badges), unsafe_allow_html=True)
    if citations:
        with st.expander("View sources"):
            st.markdown(source_cards_html(citations), unsafe_allow_html=True)


def chat_markdown() -> str:
    lines = [f"# {APP_NAME} conversation", ""]
    for m in st.session_state.messages:
        lines += [f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}", ""]
        citations = m.get("citations") or []
        lines += [f"- [{c['ref']}] {c['source']} (page {c['page']})" for c in citations]
        if citations:
            lines.append("")
    return "\n".join(lines)


def safe_documents() -> list[dict]:
    try:
        return list_documents(API_BASE_URL)
    except APIError:
        return []


# ----------------------------------------------------------------- sidebar
health = get_health(API_BASE_URL)
top_k = None  # server default; only adjustable in developer mode
with st.sidebar:
    st.markdown(f'<div class="brand">📄 {APP_NAME}</div>'
                '<div class="brand-sub">Answers from your documents, with sources</div>', unsafe_allow_html=True)

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    if health is None:
        st.warning(UNAVAILABLE_MESSAGE)
        if DEBUG:
            st.caption(f"Backend: {API_BASE_URL}")
    else:
        st.markdown('<span class="pill ok">● Ready</span>', unsafe_allow_html=True)
        if DEBUG:
            col_a, col_b = st.columns(2)
            col_a.metric("Chunks", health.get("chunks") if health.get("chunks") is not None else "-")
            col_b.metric("Documents", health.get("documents") if health.get("documents") is not None else "-")
            st.caption(f"Backend: {API_BASE_URL}")
            if health.get("model"):
                st.caption(f"Model: {health['model']}")

    if DEBUG:
        st.divider()
        top_k = st.slider("Passages to retrieve", 1, 8, 4, help="Developer setting.")

    st.divider()
    st.markdown("#### Your documents")
    if not READ_ONLY:
        uploaded = st.file_uploader(
            "Add a document (PDF, TXT or MD)", type=["pdf", "txt", "md"],
            key=f"uploader-{st.session_state.uploader_key}", label_visibility="collapsed",
        )
        if uploaded is not None and st.button("Add document", type="primary"):
            try:
                with st.spinner("Adding your document..."):
                    result = upload_document(API_BASE_URL, uploaded.name, uploaded.getvalue())
                st.session_state.flash = f"Added {result['document']}" + (f" ({result['chunks']} chunks)" if DEBUG else "")
                st.session_state.uploader_key += 1
            except APIError as exc:
                show_error(exc)
            else:
                st.rerun()

    documents = safe_documents() if health is not None else []
    if not documents and health is not None:
        st.caption("No documents yet." if READ_ONLY else "No documents yet. Add one above.")
    for doc in documents:
        extra = f"<br><span style='color:#5B7480;font-size:.78rem'>{doc['chunks']} chunks</span>" if DEBUG else ""
        if READ_ONLY:
            st.markdown(f"**{esc(doc['name'])}**{extra}", unsafe_allow_html=True)
            continue
        name_col, del_col = st.columns([5, 1])
        name_col.markdown(f"**{esc(doc['name'])}**{extra}", unsafe_allow_html=True)
        if del_col.button("✕", key=f"del-{doc['name']}", help=f"Remove {doc['name']}"):
            try:
                delete_document(API_BASE_URL, doc["name"])
                st.session_state.flash = f"Removed {doc['name']}"
            except APIError as exc:
                show_error(exc)
            else:
                st.rerun()

    if APP_PASSWORD:
        st.divider()
        if st.button("Sign out"):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.rerun()

    if st.session_state.messages:
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        st.download_button("Download conversation", data=chat_markdown(), file_name="conversation.md",
                           mime="text/markdown")

# -------------------------------------------------------------------- main
st.markdown(
    '<div class="hero"><h1>Ask your documents anything</h1>'
    '<p>Every answer comes only from your documents and shows exactly where it was found.</p>'
    '<p class="ar">اسأل بالعربية أو الإنجليزية، وستحصل على إجابة موثّقة بالمصادر.</p>'
    '<span class="tag">Answers with sources</span><span class="tag">Your documents stay private</span></div>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        render_text(message["content"])
        if message["role"] == "assistant":
            render_extras(message)

examples_slot = st.empty()
picked = None
if not st.session_state.messages:
    with examples_slot.container():
        st.markdown("##### Try one of these  ·  جرّب أحد هذه الأسئلة")
        language = st.radio(
            "Examples language", ["English", "العربية"], horizontal=True, key="example_language",
            label_visibility="collapsed",
        )
        examples = EXAMPLES_AR if language == "العربية" else EXAMPLES_EN
        columns = st.columns(2)
        for i, example in enumerate(examples):
            if columns[i % 2].button(example, key=f"example-{language}-{i}"):
                picked = example

question = st.chat_input("Ask a question about your documents  ·  اكتب سؤالك هنا...") or picked

if question:
    examples_slot.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        render_text(question)

    with st.chat_message("assistant", avatar="🤖"):
        info: dict = {}
        answer = None
        try:
            with st.spinner("Searching your documents..."):
                tokens = stream_query(API_BASE_URL, question, top_k, info)
                first = next(tokens, None)  # blocks until retrieval finished and the answer starts
            stream = itertools.chain([] if first is None else [first], tokens)
            answer = stream_rtl(stream) if has_arabic(question) else st.write_stream(stream)
        except APIError as exc:
            show_error(exc)

        if answer is not None:
            reply = {
                "role": "assistant",
                "content": answer,
                "citations": info.get("citations", []),
                "latency_ms": info.get("latency_ms"),
                "model": info.get("model"),
            }
            render_extras(reply)
            st.session_state.messages.append(reply)
