"""
Main script of the repo
"""
from transformers import pipeline

# Initialisation du générateur de texte
chatbot = pipeline('text-generation', model='distilgpt2')

# Fonction pour générer une réponse
def get_response(input_text):
    response = chatbot(input_text, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

# Exemple d'interaction
input_text = "Préfères-tu le bleu ou le rouge ?"
response = get_response(input_text)
print(response)
