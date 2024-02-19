from datasets import load_dataset, DatasetDict
import tensorflow as tf
import re
#from preprocessing.conversation import preprocess_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import re
import gc
from preprocessing.conversation import preprocess_dataset


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



tokenizer = AutoTokenizer.from_pretrained('.models/dialoGPT-fr-tokenizer')
tokenizer.pad_token = tokenizer.eos_token
# TODO : Cette façon de faire fait exploser ma mémoire il faut la revoir et uriliser au maximum les outils HF
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

def data_generator(inputs, targets, batch_size):
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        tokenized_inputs = tokenizer(batch_inputs, padding="max_length", truncation=True, max_length=512, return_tensors="tf")
        tokenized_targets = tokenizer(batch_targets, padding="max_length", truncation=True, max_length=512, return_tensors="tf")
        
        print(tokenized_inputs["input_ids"].shape, tokenized_inputs["attention_mask"].shape, tokenized_targets["input_ids"].shape)  # Ajout pour le débogage
        
        yield {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
        }, tokenized_targets["input_ids"]


# Prétraitement des données
print('Preprocessing')
train_inputs, train_targets = preprocess_dataset(raw_datasets["train"])
val_inputs, val_targets = preprocess_dataset(raw_datasets["test"])

# Tokenisation des données
print('Tokenization')
# Création du dataset TensorFlow à partir du générateur
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_inputs, train_targets, 4),
    output_signature=(
        {
            "input_ids": tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
    )
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_inputs, val_targets, 4),
    output_signature=(
        {
            "input_ids": tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
    )
).prefetch(tf.data.AUTOTUNE)




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