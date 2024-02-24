from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForCausalLM, DataCollatorWithPadding
import tensorflow as tf
import numpy as np

def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_and_prepare_data(dataset_name, tokenizer_name, subset_size=100, test_size=0.2, max_length=512):
    raw_datasets = load_dataset(dataset_name)
    train_test_split = raw_datasets["train"].select(range(subset_size)).train_test_split(test_size=test_size)
    
    raw_datasets["train"] = train_test_split["train"]
    raw_datasets["validation"] = train_test_split["test"]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    return tokenized_datasets, tokenizer

def to_tf_dataset(tokenized_datasets, tokenizer, split, batch_size=8):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    return tokenized_datasets[split].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["input_ids"],
        shuffle=True if split == "train" else False,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

def train_model(model_name, train_dataset, val_dataset, learning_rate=5e-5, epochs=5):
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    return model

def predict_and_evaluate(model, val_dataset):
    preds = model.predict(val_dataset)["logits"]
    class_preds = np.argmax(preds, axis=-1)
    print(preds.shape, class_preds.shape)

def main():
    check_gpu_availability()
    
    dataset_name = "La-matrice/french_sentences_19M"
    tokenizer_name = 'distilgpt2'
    model_name = 'distilgpt2'
    subset_size = 100
    test_size = 0.2
    max_length = 512
    batch_size = 8
    learning_rate = 5e-5
    epochs = 2
    
    tokenized_datasets, tokenizer = load_and_prepare_data(dataset_name, tokenizer_name, subset_size, test_size, max_length)
    train_dataset = to_tf_dataset(tokenized_datasets, tokenizer, "train", batch_size)
    val_dataset = to_tf_dataset(tokenized_datasets, tokenizer, "validation", batch_size)
    
    model = train_model(model_name, train_dataset, val_dataset, learning_rate, epochs)
    predict_and_evaluate(model, val_dataset)
    
    print("### Fin du script de Fine-tuning ! ###")

if __name__ == "__main__":
    main()
