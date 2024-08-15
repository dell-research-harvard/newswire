import string

import json
from statistics import mean, mode, median
from nltk.util import ngrams
import csv
import pandas as pd
import re

import transformers
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers import pipeline
from tqdm import tqdm

def clean(text):
    # clean_text = text.translate(str.maketrans('', '', r',"#$%&\()*+/:;<=>@[\\]^_`{|}~'))
    # Testing: Can we split on dashes to get the byline?
    clean_text = re.sub(r"""
               [,:;_@^%#?!&*+$<=>|{}@~`"—()]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               text, flags=re.VERBOSE)
    clean_text = clean_text.encode('ascii', 'ignore').decode()
    clean_text = clean_text.replace("..", ". ")
    clean_text = clean_text.replace(",", "")
    clean_text = clean_text.replace("\n", " ")
    clean_text = clean_text.translate(str.maketrans('', '', r'"#$%&\()*+/:;<=>@[\\]^_`{|}~,\'-'))

    # Todo: Could also remove numbers and probably more aggresive on the punctuation
    return clean_text

def most_common(text_list, threshold, total, cities=False):

    if text_list == []:
        return []
    else:
        if cities:
          percs = {k:(text_list.count(k)/total) * (1.2 ** len(k)) for k in set(text_list)}
          # percs = {k:(text_list.count(k)/total) * (len(k.split()) ** 2) for k in set(text_list)}
        else:
          percs = {k:(text_list.count(k)/total) for k in set(text_list)}

        max_value = max(percs.values())
        chosen = [key for key, value in percs.items() if value == max_value]

        if cities:
          max_value = text_list.count(chosen[0])/total

        if max_value >= threshold:
            return list(set(chosen))
        else:
            return []

def get_prop_non_words(text, spell_dict, verbose=False):
    # lightly preprocess text
    text_lower = text.lower()
    text_lower_nonum = text_lower.translate(str.maketrans('', '', digits))
    text_lower_nopunct = text_lower_nonum.translate(str.maketrans('', '', r'!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'))
    text_lower_nopunct_split = text_lower_nopunct.split()
    text_clean = text_lower_nopunct_split

    # return nwr of 1 for empty text
    if len(text_clean) == 0:
        return 1

    else:
        # perform spell checking
        dict_checked_words = [word in spell_dict.keys() for word in text_clean]

        # get proportion of correctly spelled words
        prop_correctly_spelled = sum(dict_checked_words) / len(text_clean)

        # print words found to be correctly or incorrectly spelled
        if verbose is True:
            df = pd.DataFrame({'word': text_clean, 'correctly_spelled': dict_checked_words})
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)

        return 1 - prop_correctly_spelled

def clean_ocr(text, spell_dict):
    clean_text = text.encode('ascii', 'ignore').decode()
    clean_text = clean_text.translate(str.maketrans('', '', r'"#$%&\()*+/:;<=>@[\\]^_`{|}~'))

    if len(clean_text.strip()) == 0:
        clean_text = ""
    else:

        # Clean beginnings
        first_line = [x.strip() for x in clean_text.split("\n") if x.strip()][0]
        first_line_word_list = [x.strip() for x in first_line.split() if x.strip()]
        if any(x in first_line_word_list for x in ["ee", "eee", "ae"]) or \
                (sum([len(x) for x in first_line_word_list]) / len(first_line_word_list) <= 3 and
                 get_prop_non_words(first_line, spell_dict) >= 0.5) or \
                len(first_line) <= 4:

            clean_text = clean_text[clean_text.find(first_line)+len(first_line)+1:]

        # Clean ends
        last_line = [x.strip() for x in clean_text.split("\n") if x.strip()][-1]
        last_line_word_list = [x.strip() for x in last_line.split() if x.strip()]
        if any(x in last_line_word_list for x in ["ee", "eee", "ae"]) or \
                (sum([len(x) for x in last_line_word_list]) / len(last_line_word_list) <= 3 and
                 get_prop_non_words(last_line, spell_dict) >= 0.5) or \
                len(last_line) == 1:

            clean_text = clean_text[:clean_text.rfind(last_line)]

    return clean_text

def remove_punct(ngram):
  return ngram.translate(str.maketrans('', '', string.punctuation))

def most_frequent(list):
    counter = 0
    elem = list[0]

    for i in list:
        curr_frequency = list.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            elem = i

    return (elem, counter)