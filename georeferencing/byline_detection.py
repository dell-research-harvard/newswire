import json
from glob import glob
from tqdm import tqdm
import numpy as np

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from transformers import pipeline
from datasets import Dataset

import pickle
import re


def detect_bylines(text, indices, tokenizer):
  max_index = max(indices)
  tokens = tokenizer.tokenize(text)

  all_bylines = []
  cur_byline = ""
  cur_word = ""
  byline = False
  started_word = False

  for index, token in enumerate(tokens):
    if "##" in token or "@@" in token and any(i.isalnum() for i in token):
      if (index + 1) in indices:
        byline = True
      cur_word = cur_word + re.sub("[#@]", "", token)
    else:
      if started_word:
        if byline and any(i.isalnum() for i in cur_word):
          cur_byline = cur_byline + " " + cur_word
        elif not byline and cur_byline != "":
          all_bylines.append(cur_byline.strip())
          cur_byline = ""

          if (index + 1) > max_index:
            break

        if (index + 1) in indices:
          byline = True
        else:
          byline = False

        cur_word = token
        started_word = True

      else:
        if any(i.isalnum() for i in token):
          cur_word = token
          started_word = True
        if (index + 1) in indices:
          byline = True

  if cur_byline != "":
    all_bylines.append(cur_byline.strip())

  return all_bylines

def flatten(l):
    return [item for sublist in l for item in sublist]

# Tokenize dataset
def tokenize_function(dataset):
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data02/luca_newswire/byline_model', model_max_length=128, truncation=True, padding = "max_length")
    return tokenizer(dataset['articles'], padding="max_length", truncation=True)

def get_indices(pred_indices, art_index):
  art_indices = filter(lambda x: x[0] == 0, pred_indices)
  art_indices = [x[1] for x in list(art_indices)]
  return art_indices

def batched_bylines(article_list, byline_model, tokenizer, batch_size=512):

  article_list = [" ".join(x.split(" ")[:64]) for x in article_list]

  dataset = Dataset.from_dict({'articles': article_list})
  tokenized_dataset = dataset.map(tokenize_function, batched=True)

  inference_args = TrainingArguments(output_dir="save", per_device_eval_batch_size=batch_size)
  trainer = Trainer(model=byline_model, args=inference_args)
  preds = trainer.predict(tokenized_dataset)

  token_predictions = np.argmax(preds.predictions, axis=-1)

  bylines = []

  for index, article in enumerate(token_predictions):
    pred_indices = set(flatten(np.argwhere(article==0)))
    if len(pred_indices) == 0:
      bylines.append(None)
    else:
      byline = detect_bylines(article_list[index], pred_indices, tokenizer)
      bylines.append(byline)

  return bylines