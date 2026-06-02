from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Generator

import streamlit as st

from src.agent.graph import (
    build_agent,
    extract_final_answer,
    extract_saved_order,
    extract_tool_calls,
    run_agent,
)
from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import AgentResult


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CATALOG_PATH = DATA_DIR / "products.json"
CASES_PATH = DATA_DIR / "graded_cases.json"
EXPECTED_DIR = DATA_DIR / "expected_orders"

# ─── Cấu hình trang ───────────────────────────────────────────────
st.set_page_config(
    page_title="OrderDesk AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS giống ChatGPT (light theme) ──────────────────────────────
CHATGPT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Main background ── */
    .stApp {
        background-color: #ffffff !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #f7f7f8 !important;
        border-right: 1px solid #e5e5e5;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] label {
        color: #1a1a1a !important;
    }

    /* ── Brand ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 16px;
    }
    .sidebar-brand-icon { font-size: 1.5rem; }
    .sidebar-brand-text {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: -0.3px;
    }

    /* ── Sidebar sections ── */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 600;
        color: #6e6e80 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 14px 0 6px 0;
        margin: 0;
    }

    /* ── Sidebar buttons ── */
    section[data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        border: 1px solid #d9d9e3 !important;
        color: #1a1a1a !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: background 0.15s !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #ececf1 !important;
    }

    /* ── Sidebar selectbox ── */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        border-color: #d9d9e3 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] input {
        background-color: #ffffff !important;
        border-color: #d9d9e3 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
    }

    /* ── Sidebar radio fix ── */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        padding: 8px 12px !important;
        border-radius: 8px !important;
        transition: background 0.15s ease !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: #ececf1 !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label > div > p {
        color: #353740 !important;
        font-size: 0.88rem !important;
        font-weight: 450 !important;
    }

    /* ── Sidebar expander ── */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #d9d9e3 !important;
        color: #353740 !important;
        font-size: 0.85rem !important;
    }

    /* ── Welcome ── */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 50vh;
        text-align: center;
        padding: 40px 20px;
    }
    .welcome-icon {
        width: 56px;
        height: 56px;
        background: #10a37f;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        margin-bottom: 20px;
        color: white;
    }
    .welcome-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .welcome-subtitle {
        font-size: 0.95rem;
        color: #6e6e80;
        max-width: 500px;
        line-height: 1.6;
        margin-bottom: 32px;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        max-width: 760px;
        margin: 0 auto;
        padding: 20px 0 !important;
        border-bottom: 1px solid #f0f0f0;
    }

    /* ── Chat input ── */
    .stChatInput {
        max-width: 760px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] > div {
        background-color: #f4f4f4 !important;
        border: 1px solid #d9d9e3 !important;
        border-radius: 24px !important;
        padding: 4px 8px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16,163,127,0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #1a1a1a !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8e8ea0 !important;
    }

    /* ── Status chips ── */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 6px 0;
    }
    .chip-success {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .chip-info {
        background: #f0f4ff;
        color: #4f46e5;
        border: 1px solid #c7d2fe;
    }
    .chip-error {
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }
    .chip-warn {
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
    }

    /* ── Live tool timeline ── */
    .live-timeline {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 12px 0;
    }
    .tool-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        background: #f7f7f8;
        border: 1px solid #d9d9e3;
        color: #353740;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .tool-pill-active {
        background: #ecfdf5;
        border-color: #a7f3d0;
        color: #047857;
    }
    .live-panel {
        border: 1px solid #e5e5e5;
        background: #ffffff;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 8px 0;
    }
    .live-panel-title {
        color: #6e6e80;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    /* ── Primary button (green like ChatGPT) ── */
    .stFormSubmitButton > button,
    .stButton > button[kind="primary"] {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        transition: background 0.15s !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }
    .stFormSubmitButton > button:hover,
    .stButton > button[kind="primary"]:hover {
        background: #0d8c6d !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 14px;
    }
    [data-testid="stMetricLabel"] {
        color: #6e6e80 !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e5e5;
    }

    /* ── Section title ── */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }

    /* ── Info box ── */
    .info-box {
        background: #f0fdf9;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #065f46;
        margin-bottom: 16px;
        line-height: 1.5;
    }

    /* ── Divider ── */
    .gpt-divider {
        height: 1px;
        background: #e5e5e5;
        margin: 10px 0;
        border: none;
    }

    /* ── Footer ── */
    .sidebar-footer {
        color: #8e8ea0;
        font-size: 0.72rem;
        text-align: center;
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid #e5e5e5;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #d9d9e3; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #c5c5d2; }
</style>
"""

# ─── Helpers ───────────────────────────────────────────────────────

CATEGORY_VI = {
    "laptop": "💻 Laptop",
    "monitor": "🖥️ Màn hình",
    "mouse": "🖱️ Chuột",
    "keyboard": "⌨️ Bàn phím",
    "headphone": "🎧 Tai nghe",
    "dock": "🔌 Dock",
    "storage": "💾 Ổ lưu trữ",
    "stand": "🗄️ Đế laptop",
    "webcam": "📷 Webcam",
}

CASE_CATEGORY_VI = {
    "normal": ("🟢 Bình thường", "chip-success"),
    "edge": ("🟡 Trường hợp biên", "chip-warn"),
    "clarification": ("🔵 Cần làm rõ", "chip-info"),
    "guardrail": ("🔴 Guardrail", "chip-error"),
}


def format_vnd(amount: int | float) -> str:
    return f"{int(amount):,.0f}₫".replace(",", ".")


def stock_badge(stock: int) -> str:
    if stock > 10:
        return f"🟢 {stock}"
    elif stock > 0:
        return f"🟡 {stock}"
    return "🔴 Hết"


def category_vi(cat: str) -> str:
    return CATEGORY_VI.get(cat, cat)


# ─── Data ──────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_catalog() -> list[dict[str, Any]]:
    return load_json(CATALOG_PATH)


@st.cache_data
def load_cases() -> list[dict[str, Any]]:
    return load_json(CASES_PATH)


# ─── Agent logic ───────────────────────────────────────────────────

def status_label(result: Any) -> tuple[str, str]:
    if getattr(result, "saved_order", None):
        return "✅ Đã lưu đơn hàng", "chip-success"
    return "ℹ️ Chưa lưu đơn", "chip-info"


def tool_names(result: Any) -> list[str]:
    return [tool.name for tool in getattr(result, "tool_calls", [])]


def compare_save_expectation(result: Any, case: dict[str, Any]) -> tuple[bool, str]:
    expected = case.get("expected", {})
    expect_saved = bool(expected.get("expect_saved_order", False))
    has_saved = bool(getattr(result, "saved_order", None))

    if expect_saved and has_saved:
        return True, "Kỳ vọng lưu đơn → Agent đã lưu ✓"
    if not expect_saved and not has_saved:
        return True, "Kỳ vọng không lưu → Agent không lưu ✓"
    if expect_saved and not has_saved:
        return False, "Kỳ vọng lưu đơn nhưng agent không lưu ✗"
    return False, "Kỳ vọng không lưu nhưng agent lại lưu ✗"


def run_agent_safely(query: str, provider: str, today: str):
    try:
        result = run_agent(query, provider=provider, today=today)
        return result, None
    except Exception:
        return None, traceback.format_exc()


def render_live_tool_timeline(calls: list[Any]) -> None:
    if not calls:
        st.markdown(
            '<div class="live-panel"><div class="live-panel-title">Realtime tool trace</div>'
            '<span class="chip chip-info">Chưa có tool call</span></div>',
            unsafe_allow_html=True,
        )
        return

    pills = []
    for index, call in enumerate(calls, start=1):
        active_class = " tool-pill-active" if index == len(calls) else ""
        pills.append(f'<span class="tool-pill{active_class}">{index}. {call.name}</span>')

    st.markdown(
        '<div class="live-panel"><div class="live-panel-title">Realtime tool trace</div>'
        f'<div class="live-timeline">{"".join(pills)}</div></div>',
        unsafe_allow_html=True,
    )

    latest = calls[-1]
    with st.expander(f"Tool mới nhất: {latest.name}", expanded=False):
        st.caption("Đầu vào")
        st.json(latest.args)
        st.caption("Kết quả")
        try:
            st.json(json.loads(latest.output))
        except Exception:
            st.code(latest.output or "—", language="text")


def stream_agent_realtime(
    query: str,
    provider: str,
    today: str,
    *,
    status_placeholder,
    answer_placeholder,
    trace_placeholder,
):
    started_at = time.time()
    latest_messages = []
    live_answer = ""

    try:
        status_placeholder.markdown(
            '<span class="chip chip-info">⏳ Đang khởi tạo agent...</span>',
            unsafe_allow_html=True,
        )
        answer_placeholder.markdown("💭 *Đang chuẩn bị yêu cầu...*")

        agent = build_agent(provider=provider, today=today)

        for event_index, chunk in enumerate(
            agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode=["messages", "values"],
            ),
            start=1,
        ):
            mode = None
            payload = chunk
            if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] in {"messages", "values"}:
                mode, payload = chunk

            if mode == "values" and isinstance(payload, dict) and "messages" in payload:
                latest_messages = payload["messages"]

            if mode == "messages" and isinstance(payload, tuple) and len(payload) == 2:
                message_chunk, metadata = payload
                if metadata.get("langgraph_node") == "model":
                    token = chunk_content_to_text(getattr(message_chunk, "content", ""))
                    if token:
                        live_answer += token
                        answer_placeholder.markdown(live_answer + "▌")

            elapsed = time.time() - started_at
            calls = extract_tool_calls(latest_messages)

            status_placeholder.markdown(
                (
                    '<span class="chip chip-info">'
                    f'⏳ Realtime event {event_index} · {len(calls)} tool · {elapsed:.1f}s'
                    '</span>'
                ),
                unsafe_allow_html=True,
            )

            if not live_answer:
                answer = extract_final_answer(latest_messages)
                if answer:
                    answer_placeholder.markdown(answer)
                else:
                    answer_placeholder.markdown("💭 *Đang suy nghĩ hoặc chờ tool...*")
            elif mode == "values":
                answer_placeholder.markdown(live_answer + "▌")

            trace_placeholder.empty()
            with trace_placeholder.container():
                render_live_tool_timeline(calls)

        if not latest_messages:
            raise RuntimeError("Agent did not return any messages.")

        tool_calls = extract_tool_calls(latest_messages)
        saved_order, saved_order_path = extract_saved_order(tool_calls)
        final_answer = extract_final_answer(latest_messages) or live_answer.strip()
        result = AgentResult(
            query=query,
            final_answer=final_answer,
            tool_calls=tool_calls,
            provider=provider,
            model_name=None,
            saved_order=saved_order,
            saved_order_path=saved_order_path,
        )

        label, cls = status_label(result)
        status_placeholder.markdown(f'<span class="chip {cls}">{label}</span>', unsafe_allow_html=True)
        answer_placeholder.markdown(result.final_answer or "_Không có câu trả lời cuối._")
        return result, None

    except Exception:
        status_placeholder.markdown(
            '<span class="chip chip-error">❌ Agent stream thất bại</span>',
            unsafe_allow_html=True,
        )
        return None, traceback.format_exc()


def stream_text(text: str, delay: float = 0.03) -> Generator[str, None, None]:
    """Generator streaming từng từ giống ChatGPT."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


def chunk_content_to_text(raw: Any) -> str:
    """Return streamed model text without stripping leading token spaces."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return str(raw.get("text") or raw.get("content") or "")
    if isinstance(raw, list):
        return "".join(chunk_content_to_text(item) for item in raw)
    return str(raw)


# ─── Render tool trace ─────────────────────────────────────────────

def render_tool_trace(result: Any, *, expected_tools: list[str] | None = None) -> None:
    calls = getattr(result, "tool_calls", [])

    if expected_tools is not None:
        cols = st.columns(2)
        with cols[0]:
            st.caption("🎯 Tool kỳ vọng")
            st.code(" → ".join(expected_tools) if expected_tools else "—", language="text")
        with cols[1]:
            st.caption("🔄 Tool thực tế")
            actual = tool_names(result)
            st.code(" → ".join(actual) if actual else "—", language="text")

    if calls:
        with st.expander(f"🔧 Xem chi tiết {len(calls)} bước tool", expanded=False):
            for i, record in enumerate(calls, 1):
                st.markdown(f"**Bước {i}: `{record.name}`**")
                col_in, col_out = st.columns(2)
                with col_in:
                    st.caption("📥 Đầu vào")
                    st.json(record.args)
                with col_out:
                    st.caption("📤 Kết quả")
                    try:
                        st.json(json.loads(record.output))
                    except Exception:
                        st.code(record.output or "—", language="text")
                if i < len(calls):
                    st.divider()

    saved = getattr(result, "saved_order", None)
    if saved:
        with st.expander("📦 Xem đơn hàng đã lưu", expanded=False):
            st.json(saved)
            saved_path = getattr(result, "saved_order_path", None)
            if saved_path:
                st.caption(f"📁 Lưu tại: `{saved_path}`")


# ─── Sidebar ───────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        # Brand
        st.markdown(
            '<div class="sidebar-brand">'
            '<span class="sidebar-brand-icon">🛒</span>'
            '<span class="sidebar-brand-text">OrderDesk AI</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # New chat
        if st.button("✨ Cuộc hội thoại mới", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown('<div class="gpt-divider"></div>', unsafe_allow_html=True)

        # Mode
        st.markdown('<p class="sidebar-section">Chế độ</p>', unsafe_allow_html=True)
        mode = st.radio(
            "Chế độ",
            ["💬 Chat tự do", "📋 Kịch bản test", "📦 Tra cứu sản phẩm", "🔌 Kiểm tra API"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="gpt-divider"></div>', unsafe_allow_html=True)

        # Settings
        st.markdown('<p class="sidebar-section">Cài đặt</p>', unsafe_allow_html=True)
        provider = st.selectbox("Nhà cung cấp AI", ["mimo", "google", "ollama"], index=0)
        today = st.text_input("Ngày giả lập", value="2026-06-01")

        # Footer
        st.markdown(
            '<div class="sidebar-footer">'
            'OrderDesk Agent Debugger v2.0<br>'
            'Prompt Engineering Lab'
            '</div>',
            unsafe_allow_html=True,
        )

    return mode, provider, today


# ─── Chat mode ─────────────────────────────────────────────────────

def chat_mode(provider: str, today: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome screen
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-icon">🛒</div>
                <div class="welcome-title">Tôi có thể giúp gì cho bạn?</div>
                <div class="welcome-subtitle">
                    Tạo đơn hàng, tra cứu sản phẩm, kiểm tra tồn kho
                    và áp dụng khuyến mãi tự động. Nhập yêu cầu bên dưới để bắt đầu.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggestion buttons
        col1, col2 = st.columns(2)
        suggestions = [
            ("📦 Tạo đơn hàng mới", "Tạo đơn cho Nguyễn Lan Anh, SĐT 0901234567, email lananh@example.com, giao đến 18 Nguyễn Huệ, Q.1, TP.HCM. Tôi cần 1 ASUS ROG Zephyrus G14, 2 Logitech Pebble 2 M350s và 1 LG UltraGear 27GP850-B."),
            ("🔍 Tìm laptop gaming", "Tôi cần tìm laptop gaming tầm 15-20 triệu cho sinh viên"),
            ("🎧 Đơn phụ kiện", "Mua 2 tai nghe Soundcore Q20i và 1 bàn phím Keychron K2 Max cho Trần Văn B, SĐT 0912345678, email b@email.com, giao 123 Lê Lợi, Q.1, TP.HCM"),
            ("💼 Setup văn phòng", "Tạo đơn cho công ty: 1 ThinkPad E14, 1 Dell UltraSharp U2724D, 1 Logitech MX Keys S, 1 Anker Dock cho Linh PM, email linh@corp.com, SĐT 0913002244, giao 55 Nguyễn Đình Chiểu, HN"),
        ]
        for i, (title, query) in enumerate(suggestions):
            with [col1, col2][i % 2]:
                if st.button(title, key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.rerun()

    # Display chat history
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🛒"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("result"):
                result = msg["result"]
                label, cls = status_label(result)
                st.markdown(f'<span class="chip {cls}">{label}</span>', unsafe_allow_html=True)
                render_tool_trace(result)

    # Chat input
    if prompt := st.chat_input("Nhập yêu cầu đặt hàng..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🛒"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            trace_placeholder = st.empty()

            result, error = stream_agent_realtime(
                prompt,
                provider,
                today,
                status_placeholder=status_placeholder,
                answer_placeholder=answer_placeholder,
                trace_placeholder=trace_placeholder,
            )

            if error:
                content = f"❌ Có lỗi xảy ra:\n\n```\n{error}\n```"
                answer_placeholder.markdown(content)
                st.session_state.messages.append({"role": "assistant", "content": content})
            else:
                answer = getattr(result, "final_answer", "") or "Không có câu trả lời."
                answer_placeholder.markdown(answer)
                label, cls = status_label(result)
                st.markdown(f'<span class="chip {cls}">{label}</span>', unsafe_allow_html=True)
                trace_placeholder.empty()
                render_tool_trace(result)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "result": result,
                })


# ─── Graded Cases mode ─────────────────────────────────────────────

def graded_cases_mode(provider: str, today: str) -> None:
    st.markdown('<div class="section-title">📋 Kịch Bản Có Chấm Điểm</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Chọn một kịch bản test từ danh sách, chạy agent và so sánh kết quả với kỳ vọng.</div>',
        unsafe_allow_html=True,
    )

    cases = load_cases()
    case_labels = []
    for case in cases:
        cat = case.get("category", "unknown")
        cat_label, _ = CASE_CATEGORY_VI.get(cat, (cat, "chip-info"))
        case_labels.append(f"{case['id']}  •  {cat_label}")

    selected_label = st.selectbox("Chọn kịch bản", case_labels)
    selected_index = case_labels.index(selected_label)
    case = cases[selected_index]
    expected = case.get("expected", {})

    st.text_area("📝 Nội dung yêu cầu", value=case["query"], height=100, disabled=True)

    cat = case.get("category", "unknown")
    cat_label, _ = CASE_CATEGORY_VI.get(cat, (cat, "chip-info"))
    cols = st.columns(3)
    cols[0].metric("Loại kịch bản", cat_label)
    cols[1].metric("Kỳ vọng lưu đơn", "Có ✅" if expected.get("expect_saved_order") else "Không ❌")
    cols[2].metric("Số tool yêu cầu", len(expected.get("required_tools", [])))

    expected_file = expected.get("expected_order_file")
    if expected_file:
        with st.expander("📄 Đơn hàng kỳ vọng (fixture)"):
            expected_path = ROOT_DIR / expected_file
            if expected_path.exists():
                st.json(load_json(expected_path))
            else:
                st.warning(f"Không tìm thấy: `{expected_file}`")

    if st.button("▶ Chạy kịch bản", type="primary"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        trace_placeholder = st.empty()

        result, error = stream_agent_realtime(
            case["query"],
            provider,
            today,
            status_placeholder=status_placeholder,
            answer_placeholder=answer_placeholder,
            trace_placeholder=trace_placeholder,
        )

        if error:
            st.error("❌ Thất bại!")
            st.code(error, language="text")
            return

        answer_placeholder.empty()
        trace_placeholder.empty()

        passed, message = compare_save_expectation(result, case)
        cls = "chip-success" if passed else "chip-error"
        icon = "✅" if passed else "❌"
        st.markdown(f'<span class="chip {cls}">{icon} {message}</span>', unsafe_allow_html=True)

        final_answer = getattr(result, "final_answer", "")
        if final_answer:
            st.divider()
            with st.chat_message("assistant", avatar="🛒"):
                st.markdown(final_answer)

        render_tool_trace(result, expected_tools=expected.get("required_tools", []))


# ─── Catalog mode ──────────────────────────────────────────────────

def catalog_mode() -> None:
    st.markdown('<div class="section-title">📦 Danh Mục Sản Phẩm</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Tra cứu sản phẩm trong catalog. Lọc theo danh mục, thương hiệu hoặc tìm kiếm tự do.</div>',
        unsafe_allow_html=True,
    )

    products = load_catalog()
    categories = sorted({p["category"] for p in products})
    brands = sorted({p["brand"] for p in products})

    cols = st.columns([1, 1, 2])
    category = cols[0].selectbox("📂 Danh mục", ["Tất cả", *[category_vi(c) for c in categories]])
    brand = cols[1].selectbox("🏷️ Thương hiệu", ["Tất cả", *brands])
    search = cols[2].text_input("🔍 Tìm kiếm", placeholder="Nhập tên sản phẩm...")

    vi_to_raw = {category_vi(c): c for c in categories}

    rows = []
    for p in products:
        if category != "Tất cả":
            raw_cat = vi_to_raw.get(category, category)
            if p["category"] != raw_cat:
                continue
        if brand != "Tất cả" and p["brand"] != brand:
            continue
        haystack = " ".join([p["product_id"], p["name"], p["category"], p["brand"], " ".join(p.get("tags", []))]).lower()
        if search and search.lower() not in haystack:
            continue
        rows.append({
            "Mã SP": p["product_id"],
            "Tên sản phẩm": p["name"],
            "Danh mục": category_vi(p["category"]),
            "Thương hiệu": p["brand"],
            "Giá (VNĐ)": format_vnd(p["unit_price"]),
            "Tồn kho": stock_badge(p["stock"]),
            "Tags": ", ".join(p.get("tags", [])),
        })

    st.caption(f"Hiển thị **{len(rows)}** / **{len(products)}** sản phẩm")
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ─── API Check mode ───────────────────────────────────────────────

def api_check_mode() -> None:
    st.markdown('<div class="section-title">🔌 Kiểm Tra Kết Nối API</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Gọi thử API Mimo để xác nhận kết nối hoạt động bình thường.</div>',
        unsafe_allow_html=True,
    )

    if st.button("Kiểm tra Mimo API", type="primary"):
        try:
            with st.spinner("Đang kết nối..."):
                model = build_chat_model(provider="mimo", temperature=0.0)
                response = model.invoke("Tra loi dung mot tu: OK")
                text = normalize_content(response.content)
            st.markdown('<span class="chip chip-success">✅ Kết nối thành công</span>', unsafe_allow_html=True)
            st.code(text, language="text")
        except Exception:
            st.markdown('<span class="chip chip-error">❌ Kết nối thất bại</span>', unsafe_allow_html=True)
            st.code(traceback.format_exc(), language="text")


# ─── Main ──────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(CHATGPT_CSS, unsafe_allow_html=True)
    mode, provider, today = render_sidebar()

    if mode == "💬 Chat tự do":
        chat_mode(provider, today)
    elif mode == "📋 Kịch bản test":
        graded_cases_mode(provider, today)
    elif mode == "📦 Tra cứu sản phẩm":
        catalog_mode()
    elif mode == "🔌 Kiểm tra API":
        api_check_mode()


if __name__ == "__main__":
    main()
