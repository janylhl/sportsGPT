import re

def preprocess_conversation(exchange):
    splits = re.split(r"(\[speaker\d{3}:\])", exchange)
    dialogue = [segment for segment in splits if segment.strip()]  # Filtrer les segments vides
    
    inputs, targets = [], []
    for i in range(len(dialogue)-2):
        if "[speaker001:]" in dialogue[i+1]:
            input_text = dialogue[i] + dialogue[i+1]
            target_text = dialogue[i+1] + dialogue[i+2]
            inputs.append(input_text.strip())
            targets.append(target_text.strip())
    
    return inputs, targets

def preprocess_dataset(dataset):
    processed_inputs, processed_targets = [], []
    for conversation in dataset:
        inputs, targets = preprocess_conversation(conversation["text"])  # Adaptez selon la structure de votre dataset
        processed_inputs.extend(inputs)
        processed_targets.extend(targets)
    return processed_inputs, processed_targets