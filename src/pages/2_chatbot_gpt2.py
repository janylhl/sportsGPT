import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from llm.response import get_response_gpt
from transformers import pipeline

# Initialisation du pipeline de génération de texte
generator = pipeline('text-generation', model='gpt2')  # Remplacez 'gpt2' par 'EleutherAI/gpt-neo-2.7B' ou tout autre modèle si nécessaire



st.title("💬 FitBot")
st.write("This version is broken for the moment, prefer the dialoGPT version.")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container

    print(st.session_state.messages)
    response, history = get_response_gpt(st.session_state.messages, generator)
    #msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
