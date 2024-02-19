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

# Load custom generator for french DialoGPT
tokenizer = AutoTokenizer.from_pretrained('.models/dialoGPT-fr-tokenizer')
tokenizer.pad_token = tokenizer.eos_token


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


batch_size = 8
num_epochs = 3
# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_train_steps = len(train_dataset) * num_epochs
lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps)
opt = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
model.fit(train_dataset, validation_data=val_dataset, epochs=3)



import evaluate
import numpy as np
preds = model.predict(val_dataset)["logits"]
class_preds = np.argmax(preds, axis=1)
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=class_preds, references=raw_datasets["validation"]["label"])


from transformers import TFConversationalPipeline
# Création d'un pipeline conversationnel
chatbot = TFConversationalPipeline(model=model, tokenizer=tokenizer)
# Générer une réponse
inputs = tokenizer.encode("Bonjour, comment ça va ?", return_tensors='tf')
outputs = model.generate(inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))