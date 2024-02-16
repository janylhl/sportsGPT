import streamlit as st
from transformers import pipeline

# Initialisation du modèle de chatbot
chatbot = pipeline('text-generation', model='distilgpt2')

def get_response(input_text):
    messages_str = " ".join([msg["content"] for msg in input_text if msg["role"] == "user"])
    responses = chatbot(messages_str, max_length=50, num_return_sequences=1)
    return responses[0]['generated_text']

st.title("💬 SportGPT")

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
    response = get_response(st.session_state.messages)
    #msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
