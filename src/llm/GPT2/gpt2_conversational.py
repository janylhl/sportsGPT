from transformers import pipeline

# Initialisation du pipeline de génération de texte
generator = pipeline('text-generation', model='gpt2')  # Remplacez 'gpt2' par 'EleutherAI/gpt-neo-2.7B' ou tout autre modèle si nécessaire

def get_response(input_text, history=[]):
    # Construction du contexte de conversation
    conversation_context = "Vous êtes un chatbot intelligent. "
    for msg in history:
        conversation_context += f"Utilisateur: {msg['user']} Chatbot: {msg['bot']} "
    conversation_context += f"Utilisateur: {input_text} Chatbot: "
    
    # Génération de la réponse
    responses = generator(conversation_context, max_length=1000, num_return_sequences=1)
    generated_text = responses[0]['generated_text']
    
    # Extraire uniquement la dernière réponse générée
    response = generated_text.split("Chatbot: ")[-1].split(" Utilisateur: ")[0]
    
    # Mise à jour de l'historique de la conversation
    history.append({"user": input_text, "bot": response})
    
    return response, history

# Exemple d'utilisation
#input_text = "Quelle est la météo aujourd'hui ?"
#response, history = get_response(input_text)
#print("Réponse du chatbot :", response)
