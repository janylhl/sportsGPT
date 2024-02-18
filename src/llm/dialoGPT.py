from transformers import pipeline, Conversation

# Initialisation du modèle de chatbot avec DialoGPT
chatbot = pipeline('conversational', model='microsoft/DialoGPT-medium')

def get_response(input_text):
    # Création d'une conversation
    conversation = Conversation(input_text)
    
    # Génération de la réponse du chatbot
    chatbot(conversation)
    
    # Récupération de la dernière réponse générée
    response = conversation.generated_responses[-1]
    return response
