from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import tensorflow as tf

# Chargement de l'ensemble de données
raw_datasets = load_dataset("esoria3/french_simplified")
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Définir le token de padding sur le token eos si ce n'est pas déjà fait
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Fonction de tokenisation ajustée
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenisation de l'ensemble de données
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Préparation du collateur de données
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# Conversion en dataset TensorFlow
def to_tf_dataset(split):
    return tokenized_datasets[split].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["input_ids"],  # Les labels sont les input_ids pour un modèle de langage
        shuffle=True if split == "train" else False,
        collate_fn=data_collator,
        batch_size=8,
    )

tf_train_dataset = to_tf_dataset("train")
tf_validation_dataset = to_tf_dataset("validation")

# Chargement et compilation du modèle
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Entraînement du modèle
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)

# Prédiction et évaluation
preds = model.predict(tf_validation_dataset)["logits"]
class_preds = np.argmax(preds, axis=-1)  # Ajustez selon la structure de votre ensemble de données
print(preds.shape, class_preds.shape)
