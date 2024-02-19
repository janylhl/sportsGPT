from datasets import load_dataset
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import re
import gc
from preprocessing.conversation import preprocess_dataset  # Assurez-vous que ce module est correctement défini

# Vérification de la disponibilité du GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



def preprocess_and_tokenize_datasets(dataset_name: str, tokenizer_path: str, batch_size: int = 4) -> tuple:
    """
    Charge, prétraite, et tokenize les datasets d'entraînement et de validation.
    
    Args:
        dataset_name (str): Nom du dataset à charger.
        tokenizer_path (str): Chemin vers le tokenizer personnalisé.
        batch_size (int): Taille des lots pour le traitement.
    
    Returns:
        tuple: Contenant les datasets d'entraînement et de validation tokenisés.
    """
    print('Chargement et prétraitement du dataset')
    raw_datasets = load_dataset(dataset_name)
    train_inputs, train_targets = preprocess_dataset(raw_datasets["train"])
    val_inputs, val_targets = preprocess_dataset(raw_datasets["test"])

    # Calcul du nombre d'étapes par époque pour l'entraînement
    steps_per_epoch_train = len(train_inputs) // batch_size

    # Calcul du nombre d'étapes par époque pour la validation
    validation_steps = len(val_inputs) // batch_size

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_tf_dataset_from_generator(tokenizer, train_inputs, train_targets, batch_size)
    val_dataset = create_tf_dataset_from_generator(tokenizer, val_inputs, val_targets, batch_size)

    return train_dataset, val_dataset, tokenizer, steps_per_epoch_train, validation_steps

def create_tf_dataset_from_generator(tokenizer, inputs, targets, batch_size) -> tf.data.Dataset:
    """
    Crée un dataset TensorFlow à partir d'un générateur qui tokenise dynamiquement les données.

    Args:
        tokenizer: Tokenizer Hugging Face chargé.
        inputs (list): Liste des inputs à tokenizer.
        targets (list): Liste des targets à tokenizer.
        batch_size (int): Taille des lots pour le traitement.
    
    Returns:
        tf.data.Dataset: Dataset TensorFlow prêt pour l'entraînement/la validation.
    """
    def data_generator():
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            tokenized_inputs = tokenizer(batch_inputs, padding="max_length", truncation=True, max_length=512, return_tensors="tf")
            tokenized_targets = tokenizer(batch_targets, padding="max_length", truncation=True, max_length=512, return_tensors="tf")
            yield {"input_ids": tokenized_inputs["input_ids"], "attention_mask": tokenized_inputs["attention_mask"]}, tokenized_targets["input_ids"]

    return tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            {
                "input_ids": tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
        )
    ).prefetch(tf.data.AUTOTUNE)

def train_and_evaluate_model(train_dataset, val_dataset, model_name: str, num_epochs: int = 1, steps_per_epoch_train: int = 1, validation_steps: int = 1):
    """
    Entraîne et évalue un modèle sur les datasets fournis.
    
    Args:
        train_dataset (tf.data.Dataset): Dataset d'entraînement.
        val_dataset (tf.data.Dataset): Dataset de validation.
        model_name (str): Nom du modèle à charger pour l'entraînement.
        num_epochs (int): Nombre d'époques pour l'entraînement.
    """
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    print("Entraînement")


    model.fit(train_dataset, validation_data=val_dataset, epochs=3, steps_per_epoch=steps_per_epoch_train, validation_steps=validation_steps)

    # TODO : Ajoutez ici votre logique d'évaluation si nécessaire

    return model


def generate_dialogue_response(model, tokenizer, input_text: str):
    """
    Génère une réponse de dialogue à partir du modèle entraîné.
    
    Args:
        model: Modèle entraîné pour la génération de texte.
        tokenizer: Tokenizer utilisé avec le modèle.
        input_text (str): Texte d'entrée pour lequel générer une réponse.
    
    Returns:
        str: Réponse générée par le modèle.
    """
    inputs = tokenizer.encode(input_text, return_tensors='tf')
    outputs = model.generate(inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    DATASET_NAME = "OpenLLM-France/Claire-Dialogue-French-0.1"
    MODEL_NAME = "microsoft/DialoGPT-small"
    # Chemin vers le tokenizer personnalisé
    TOKENIZER_PATH = '.models/dialoGPT-fr-tokenizer'
    # Prétraitement, tokenisation, et création des datasets
    train_dataset, val_dataset, tokenizer, steps_per_epoch_train, validation_steps = preprocess_and_tokenize_datasets(DATASET_NAME, TOKENIZER_PATH)

    # Entraînement et évaluation du modèle
    model = train_and_evaluate_model(train_dataset, val_dataset, MODEL_NAME, steps_per_epoch_train, validation_steps)

    # Génération d'une réponse de dialogue
    chat_response = generate_dialogue_response(model, tokenizer, "Bonjour, comment ça va ?")
    print(chat_response)

