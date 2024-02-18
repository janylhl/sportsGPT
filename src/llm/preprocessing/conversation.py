import re

def preprocess_conversation(exchange):
    splits = re.split(r"(\[speaker\d{3}:\])", exchange)
    dialogue = [segment.strip() for segment in splits if segment.strip()]  # Nettoyage des segments
    
    inputs, targets = [], []
    # Boucle ajustée pour mieux capturer le contexte
    for i in range(1, len(dialogue)-1, 2):
        context = " ".join(dialogue[max(0, i-3):i+1])  # Inclure plus de contexte si possible
        input_text = context
        target_text = dialogue[i+1]
        inputs.append(input_text)
        targets.append(target_text)
    
    return inputs, targets


def preprocess_dataset(dataset):
    processed_inputs, processed_targets = [], []
    for conversation in dataset:
        inputs, targets = preprocess_conversation(conversation["text"])  # Adaptez selon la structure de votre dataset
        processed_inputs.extend(inputs)
        processed_targets.extend(targets)
    return processed_inputs, processed_targets