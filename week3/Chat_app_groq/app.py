import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Page setup
st.set_page_config(page_title="Groq Chat", layout="centered")
st.title("💬 Chat with Groq LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_resp = ""

       
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=st.session_state.messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_resp += delta
            placeholder.markdown(full_resp + "▌")

        placeholder.markdown(full_resp)

    st.session_state.messages.append({"role": "assistant", "content": full_resp})

