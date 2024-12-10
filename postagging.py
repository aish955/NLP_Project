import numpy as np
import seaborn as sns

import torch

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(train_df.head())

target_map = {}
test_target = {}
i = 0
for source in train_df['label'].unique():
  target_map[source] = i
  test_target[source] = i
  i+=1

from datasets import load_dataset
from datasets import Dataset, DatasetDict

train_df['target'] = train_df['label'].map(target_map)
test_df['target'] = test_df['label'].map(test_target)

train_df = train_df[['Generation','target']]
train_df.columns = ['text','label']


test_df = test_df[['Generation','target']]
test_df.columns = ['text','label']
train_dataset = Dataset.from_dict(train_df)
test_dataset = Dataset.from_dict(test_df)

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def pos_tagging(example):
    
    tokens = word_tokenize(example['text'])
    
    tagged = pos_tag(tokens)
   
    example['pos_tags'] = [tag for word, tag in tagged]
    return example

dataset1 = train_dataset.map(pos_tagging)
dataset1.to_csv("pos_tagged_output.csv", index=False)

def remove_adjectives(example):
    tokens = word_tokenize(example['text'])
    
    tagged = pos_tag(tokens)
   
    filtered_tokens = [word for word, tag in tagged if tag not in ['JJ', 'JJR', 'JJS']]
  
    example['text'] = " ".join(filtered_tokens)
    return example

new_dataset = dataset1.map(remove_adjectives, batched=False)

new_dataset = new_dataset.remove_columns(["pos_tags"])
new_dataset.to_csv("adjectives_removed.csv", index=False)
