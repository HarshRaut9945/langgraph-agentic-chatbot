import streamlit as st
from langchain_core.messages import HumanMessage
from agentic_chatbot_backend import chatbot

# -----------------------
# Page Config
# -----------------------
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
    st.session_state.thread_id = "thread-1"

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# -----------------------
# Display Chat History
# -----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------
# Streaming Generator
# -----------------------
def stream_response(prompt):
    """
    Streams the AI response token by token from LangGraph.
    """
    for message_chunk, metadata in chatbot.stream(
        {
            "messages": [
                HumanMessage(content=prompt)
            ]
        },
        config=config,
        stream_mode="messages"
    ):
        if message_chunk.content:
            yield message_chunk.content


# -----------------------
# User Input
# -----------------------
if prompt := st.chat_input("Ask anything..."):

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display streaming assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(prompt))

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )