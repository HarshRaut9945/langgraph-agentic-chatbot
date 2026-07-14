import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agentic_chatbot_backend import chatbot

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph Chatbot")
st.write("Powered by Gemini + LangGraph")

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# -----------------------
# Display Previous Messages
# -----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------
# User Input
# -----------------------
if prompt := st.chat_input("Ask anything..."):

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call LangGraph
    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=prompt)
            ]
        },
        config=config
    )

    answer = response["messages"][-1].content

    # Show assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)