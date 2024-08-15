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

model = AutoModelForTokenClassification.from_pretrained('byline_entity_model').to('cuda')
tokenizer = AutoTokenizer.from_pretrained('byline_entity_model', model_max_length=64, truncation=True, padding = "max_length")
pipe = pipeline('ner', model=model, tokenizer=tokenizer, device=0, aggregation_strategy="first")

def detect_entities(text, preds):
  rem_text = ""
  locs, col_indices = [], []

  for pred in preds:
    if pred['entity_group'] == "location":
      locs.append(pred['word'].strip())
    elif pred['entity_group'] == "columnist":
      col_indices.extend([x for x in range(pred['start'], pred['end'] + 1)])
    else:
      rem_text = rem_text + pred['word'].strip()

  for index, char in enumerate(text):
    if index not in col_indices:
      rem_text = rem_text + char

  return locs, rem_text


def remove_columnists(bylines):
  preds = pipe(bylines, batch_size=256)

  cleaned_bylines = []
  for idx, _ in tqdm(enumerate(bylines)):
      _, rem_text = detect_entities(bylines[idx], preds[idx])
      cleaned_bylines.append(rem_text)

  return cleaned_bylines