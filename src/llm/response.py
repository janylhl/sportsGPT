import tensorflow as tf

from transformers import Conversation
def get_response_gpt2(input_text, tokenizer, model):
    input_strings = [item['content'] for item in input_text]

    input_tokenized = tokenizer.encode(input_strings, add_special_tokens=False)
    input_ids = tf.keras.preprocessing.sequence.pad_sequences([input_tokenized], maxlen=521, padding="post")
    output_ids = model.generate(input_ids, max_length=512, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text



def get_response_chatbot(input_text, chatbot):
    # Création d'une conversation
    conversation = Conversation(input_text)
    
    # Génération de la réponse du chatbot
    chatbot(conversation)
    
    # Récupération de la dernière réponse générée
    response = conversation.generated_responses[-1]
    return response

def get_response_gpt(input_text, generator, history=[]):
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
