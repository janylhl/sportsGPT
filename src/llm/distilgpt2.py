from transformers import pipeline

# Initialisation du modèle de chatbot
chatbot = pipeline('text-generation', model='distilgpt2')

def get_response(input_text):
    messages_str = " ".join([msg["content"] for msg in input_text if msg["role"] == "user"])
    responses = chatbot(messages_str, max_length=100, num_return_sequences=1)
    return responses[0]['generated_text']