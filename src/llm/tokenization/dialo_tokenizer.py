from datasets import load_dataset
from transformers import AutoTokenizer

# Corpus d'entrainement français
raw_datasets = load_dataset("La-matrice/french_sentences_19M")



def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = get_training_corpus()


# Charger le tokenizer DialoGPT original
old_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
# Entrainer le nouveau tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
# Enregistrer le nouveau tokenizer
tokenizer.save_pretrained('.models/dialoGPT-fr-tokenizer')
