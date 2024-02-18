from datasets import load_dataset, DatasetDict
import tensorflow as tf
import re
#from preprocessing.conversation import preprocess_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import re
import gc

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

# Checking GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print('Chargement du dataset')
# Chargement du dataset
raw_datasets = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1")

# Imprimer des échantillons
#for input, target in zip(val_inputs[:5], val_targets[:5]):  # Imprimer les 5 premières paires
#    print("Input:", input)
#    print("Target:", target)
#    print("---")



tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token

def batch_tokenize_pairs(inputs, targets, batch_size=10):
    tokenized_inputs = []
    tokenized_targets = []
    
    for i in range(0, len(inputs), batch_size):
        try:
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            batch_tokenized_inputs = tokenizer(batch_inputs, padding=True, truncation=True, max_length=512, return_tensors="tf")
            batch_tokenized_targets = tokenizer(batch_targets, padding=True, truncation=True, max_length=512, return_tensors="tf")
            
            # Supposons que vous voulez juste stocker les input_ids pour simplifier
            tokenized_inputs.extend(batch_tokenized_inputs['input_ids'])
            tokenized_targets.extend(batch_tokenized_targets['input_ids'])
            gc.collect()
        except Exception as e:
            print(f"Erreur lors de la tokenisation du lot {i}-{i+batch_size}: {e}")
            break  # ou continue, selon si vous voulez arrêter au premier échec ou essayer de continuer

    return tokenized_inputs, tokenized_targets

def create_tf_dataset(tokenized_inputs, tokenized_targets):
    # Extraire input_ids et attention_mask des inputs tokenisés
    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']
    
    # Les labels sont les input_ids des targets tokenisés
    labels = tokenized_targets['input_ids']
    
    # Créer un tf.data.Dataset à partir des tensors
    dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, labels))
    
    # Mélanger le dataset et le diviser en lots
    dataset = dataset.shuffle(len(input_ids)).batch(8)
    
    return dataset


# Prétraitement des données
print('Preprocessing')
train_inputs, train_targets = preprocess_dataset(raw_datasets["train"])
val_inputs, val_targets = preprocess_dataset(raw_datasets["test"])

# Tokenisation des données
print('Tokenization')
print('Tokenization train')
tokenized_train_inputs, tokenized_train_targets = batch_tokenize_pairs(train_inputs, train_targets)
print('Tokenization val')
tokenized_val_inputs, tokenized_val_targets = batch_tokenize_pairs(val_inputs, val_targets)

# Création des datasets TensorFlow
print('Dataset creation')
train_dataset = create_tf_dataset(tokenized_train_inputs['input_ids'], tokenized_train_targets['input_ids'])
val_dataset = create_tf_dataset(tokenized_val_inputs['input_ids'], tokenized_val_targets['input_ids'])


from transformers import TFAutoModelForCausalLM

model = TFAutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
print("Entrainement")
# Compilation du modèle
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# Entraînement
model.fit(train_dataset, validation_data=val_dataset, epochs=3)



from transformers import TFConversationalPipeline

# Création d'un pipeline conversationnel
chatbot = TFConversationalPipeline(model=model, tokenizer=tokenizer)

# Générer une réponse
inputs = tokenizer.encode("Bonjour, comment ça va ?", return_tensors='tf')
outputs = model.generate(inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))