import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_graph
from extractor import extract_and_save_facts
from memory import save_message

load_dotenv()

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")

# Session state init
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_001"          
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

with st.sidebar:
    st.markdown("### ession Info")
    st.code(f"Thread: {st.session_state.thread_id[:16]}…")
    st.code(f"User:   {st.session_state.user_id}")

    if st.button("New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages  = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**Tools available**\n"
        "- 🌤️ Weather (OpenWeatherMap)\n"
        "- 🔍 Web search (Tavily)\n"
        "- 🧠 Long-term memory (MongoDB)"
    )

# Render existing chat history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and agent invocation
if prompt := st.chat_input("Ask me anything…"):

    # 1. Show and persist the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    save_message(st.session_state.thread_id, "human", prompt)

    # 2. Invoke the LangGraph agent
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "user_id": st.session_state.user_id,
        }
    }

    with st.spinner("Thinking…"):
        result = st.session_state.graph.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "user_id": st.session_state.user_id,
            },
            config = config
        )

    ai_reply = result["messages"][-1].content

    # 3. Show and persist the AI reply
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
    save_message(st.session_state.thread_id, "ai", ai_reply)

    # 4. Background fact extraction from MongoDB
    extract_and_save_facts(st.session_state.user_id, prompt, ai_reply)
