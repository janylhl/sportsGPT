import streamlit as st
from transformers import pipeline

# Initialisation du modèle de chatbot
chatbot = pipeline('text-generation', model='distilgpt2')

def get_response(input_text):
    responses = chatbot(input_text, max_length=50, num_return_sequences=1)
    return responses[0]['generated_text']

st.title("💬 Chatbot")

# Initialisation ou récupération de l'historique des messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Affichage de l'historique des messages
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message(message["content"], is_user=True)
    else:
        st.chat_message(message["content"], is_user=False)

# Entrée de l'utilisateur
user_input = st.text_input("You:", key="user_input")

# Envoi et traitement de la réponse
if st.button("Send"):
    if user_input:
        # Ajout du message de l'utilisateur à l'historique
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        # Génération de la réponse du modèle
        response = get_response(user_input)
        
        # Ajout de la réponse du modèle à l'historique
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Mise à jour de l'interface pour afficher les nouveaux messages
        st.experimental_rerun()
