import os
import pandas as pd
from transformers import BertTokenizer

def load_data():
    base_path = "data/TuringBench/AA"
    train_path = os.path.join(base_path, "train.csv")
    valid_path = os.path.join(base_path, "valid.csv")
    test_path = os.path.join(base_path, "test.csv")

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)

    print("Data loaded successfully!")
    return train_data, valid_data, test_data

def tokenize_data(data, tokenizer, max_length=128):
    tokens = data["Generation"].apply(
        lambda x: tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    )
    
    data["input_ids"] = tokens.apply(lambda x: x["input_ids"].squeeze().tolist())
    data["attention_mask"] = tokens.apply(lambda x: x["attention_mask"].squeeze().tolist())

    return data[["input_ids", "attention_mask", "label"]]

if __name__ == "__main__":
    train_data, valid_data, test_data = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_data = tokenize_data(train_data, tokenizer)
    valid_data = tokenize_data(valid_data, tokenizer)
    test_data = tokenize_data(test_data, tokenizer)

    # Save tokenized data as lists (not tensors)
    train_data.to_csv("data/tokenized_train.csv", index=False)
    valid_data.to_csv("data/tokenized_valid.csv", index=False)
    test_data.to_csv("data/tokenized_test.csv", index=False)

    print("Tokenized datasets saved with labels!")
