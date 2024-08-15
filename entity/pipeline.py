import pickle 

##Import tokenizer from HF
from transformers import AutoTokenizer
from tqdm import tqdm
from glob import glob
import os
from data_fns import featurise_data 
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer
import sys
import numpy as np
import pandas as pd
import json
import argparse
import faiss
# print(data.keys())

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)

from utils.utils import get_cluster_assignment


class TokenizerInfoSlow():
    """
    Given string of text, creates an object containing information on
    the characters, tokens, and words of that text;
    this class is used for slow tokenizers, which do not have some
    of this functionality already written
    - A word is defined to be any consecutive sequence of alphanumeric characters
      or any single non-alphanumeric and non-whitespace character
    - All newlines are converted to spaces before running tokenizer to ensure
      they are not included in the tokens; however, the method text() still
      returns the original text
    - The special classifcation and separator tokens (CLS and SEP) are added at
      the beginning and end of the token sequence, respectively, as is expected
      by BERT-based models
    """

    @staticmethod
    def __token_len(token):
      """
      Given a token as a string from the tokenizer,
      returns the number of characters in that token after undoing the
      modifications that the tokenizer makes to the token, so that only
      the characters originally present in the text are counted
      """
      # Note that the below only handles the slow tokenizer used for BERTweet, since
      # the rest of the models tested use fast tokenizers and therefore use
      # the class TokenizerInfoFast
      # Other slow tokenizers will need different ways of finding token length
      return 1 if token == tokenizer.unk_token else len(token.replace('@@', ''))

    def __init__(self, text):
      # (The purpose of each instance variable is described in the subsequent methods in this class)
      self.__text = text
      # Ensure tokenizer ignores newlines
      text = text.replace('\n', ' ')
      # Number of characters in each token; None for CLS and SEP tokens at beginning and end
      token_lengths = [None] + [self.__token_len(t) for t in tokenizer.tokenize(text)] + [None]
      # Number of tokens that make up the text
      token_count = len(token_lengths)
      self.__input_ids = tokenizer(text)['input_ids']
      self.__word_ids = [[] for _ in range(token_count)]
      self.__word_to_chars = []
      self.__char_to_word = []
      self.__token_to_chars = [None] * token_count
      self.__char_to_token = [None] * len(text)

      start_char_id = 0  # Index of starting character in current word
      word_id = 0  # Index of current word
      is_word_end = [False] * len(text)  # For each character in text, whether that character is the final character of a word
      text_with_final_space = text + ' '  # Extra space added to end to ensure final word is properly processed
      final_index = len(text)  # Index of the extra space
      for i, c in enumerate(text_with_final_space):
        if c.isalnum():  # Alphanumeric character -- indicates we are in the middle of a word
          # Record index of word that the current character is in
          self.__char_to_word.append(word_id)

        else:  # Non-alphanumeric character -- indicates word boundary
          if i > start_char_id:  # Indicates there was a word being processed before current character
            # Since we have reached word boundary, the previous character finished the word
            is_word_end[i-1] = True
            # Record indices of starting and ending characters of that word
            self.__word_to_chars.append((start_char_id, i))
            # Update word index for current character
            word_id += 1

          if i == final_index:  # Reached end -- don't process the extra space
            self.__word_count = word_id
            break

          # Record index of word that the current character is in; None if character is whitespace
          self.__char_to_word.append(word_id if not c.isspace() else None)

          if not c.isspace():  # Current character is neither alphanumeric nor whitespace (punctuation, etc.)
            # The current character constitutes a single-character word
            is_word_end[i] = True
            # Record current character as its own word
            self.__word_to_chars.append((i, i+1))
            # Update word index for subsequent character
            word_id += 1

          # Set starting character of subsequent word
          start_char_id = i + 1

      start_char_id = 0  # Index of starting charcter in current token
      token_char_count = 0  # Number of characters already processed in current token
      token_id = 1  # Index of the current token -- starts at 1 to account for CLS at index 0
      token_char_goal = token_lengths[token_id]  # Total number of characters in current token
      last_token_id = token_count - 2  # Index of the last token before SEP
      for i, c in enumerate(text):
        # Ignore whitespace
        if c.isspace():
          start_char_id = i + 1
          continue
        self.__char_to_token[i] = token_id
        token_char_count += 1
        # Reached character that ends either a token or a word
        if token_char_count == token_char_goal or is_word_end[i]:
          # Add association between token index and word index
          self.__word_ids[token_id].append(self.__char_to_word[i])
          # If end of token, update/reset token properties
          if token_char_count == token_char_goal:
            self.__token_to_chars[token_id] = (start_char_id, i+1)
            if token_id == last_token_id:
              break
            start_char_id = i + 1
            token_id += 1
            token_char_count = 0
            token_char_goal = token_lengths[token_id]

    def text(self):
      """
      Returns the original string of text that was given when instantiating this class
      """
      return self.__text

    def input_ids(self):
      """
      Returns an integer list containing the numerical representations of
      each token when the text is run through the tokenizer;
      these are the token represenations that are passed as input to the NER model
      """
      return self.__input_ids

    def word_ids(self):
      """
      Returns list that indicates, for each token index,
      the indices of all the words that contain that token
      (so that each element of the list is itself a list)
      """
      return self.__word_ids

    def word_to_chars(self, i):
      """
      Given word index, returns tuple containing the indices
      of the starting and ending characters of that word
      (with start included and end excluded, as is standard)
      """
      return self.__word_to_chars[i]

    def char_to_word(self, i):
      """
      Given character index, returns index of word that contains that character;
      None when character not in any word
      """
      return self.__char_to_word[i]

    def token_to_chars(self, i):
      """
      Given token index, returns tuple containing the indices
      of the starting and ending characters of that token
      (with start included and end excluded, as is standard)
      """
      return self.__token_to_chars[i]

    def char_to_token(self, i):
      """
      Given character index, returns index of token that contains that character;
      None when token not in any word
      """
      return self.__char_to_token[i]
    
    def word_count(self):
      """
      Returns number of words in text
      """
      return self.__word_count


class TokenizerInfoFast():
    """
    Same as TokenizerInfoSlow, but for fast tokenizers
    - A word is defined however the specified tokenizer defines it
    - word_to_chars() and token_to_chars() return a CharSpan object rather
      than a tuple, but this object is subscriptable in the same way as a tuple
    """
    def __init__(self, text):
      self.__text = text
      self.__inputs = tokenizer(text.replace('\n', ' '))

    def text(self):
      return self.__text

    def input_ids(self):
      return self.__inputs['input_ids']

    def word_ids(self):
      return [[i] for i in self.__inputs.word_ids()]

    def word_to_chars(self, i):
      return self.__inputs.word_to_chars(i)

    def char_to_word(self, i):
      return self.__inputs.char_to_word(i)

    def token_to_chars(self, i):
      return self.__inputs.token_to_chars(i)

    def char_to_token(self, i):
      return self.__inputs.char_to_token(i)
    
    def word_count(self):
      return 1 + next((elt for elt in reversed(self.__inputs.word_ids()) if elt is not None), -1)


def clean_ocr_text(text, basic=True, remove_list=["#","/","*","@","~","¢","©","®","°"]):
    """
    Given 
    - string of text,
    - whether (True/False) to do only basic newline cleaning, and
    - the list of characters to remove (if basic=False),
    returns a tuple containing
    (1) the text after applying the desired cleaning operations, and
    (2) a list of integers indicating, for each character in original text,
        how many positions to the left that character is offset to arrive at cleaned text.
    When basic is False, also replaces 'é', 'ï', 'ﬁ', and 'ﬂ'.
    In all cases, hyphen-newline ("-\n") sequences are removed, lone newlines are
    converted to spaces, and sequences of consecutive newlines are kept unchanged
    in order to indicate paragraph boundaries.
    """
    # Code to deal with unwanted symbols
    cleaned_text = text.replace("-\n", "")
    if not basic:
      cleaned_text = cleaned_text.replace("é", "e").replace("ï", "i").replace("ﬁ", "fi").replace("ﬂ", "fl")
      cleaned_text = cleaned_text.translate({ord(x): '' for x in remove_list})
      
    # Code to deal with newline and double newline
    z = 0
    while z < (len(cleaned_text)-1):  # Check from the first to before last index
          if cleaned_text[z] == "\n" and cleaned_text[z+1] == "\n":
              z += 2
          elif cleaned_text[z] == "\n" and cleaned_text[z+1] != "\n":
              temp = list(cleaned_text)
              temp[z] = " "
              cleaned_text = "".join(temp)
              z += 1
          else:
              z += 1
    if cleaned_text[len(cleaned_text)-1] == "\n" and cleaned_text[len(cleaned_text)-2] != "\n":  # Check if the last index is a new line
      temp = list(cleaned_text)
      temp[len(cleaned_text)-1] = " "
      cleaned_text = "".join(temp)  

    # Code to adjust offsets  
    offsets = []
    cur_offset = 0
    i = 0
      
    while i < len(text):
      if i+1 < len(text) and text[i:i+2] == '-\n':  # Found removed hyphen-newline
        offsets.extend([cur_offset, cur_offset + 1])  # Make removed characters correspond with next character
        cur_offset += 2  # Update offset for subsequent characters
        i += 2  # Push forward beyond removed characters
      else:
        offsets.append(cur_offset)  # Record offset of non-removed character
        i += 1  # Process next character
    
    if not basic:            
      for j in range(len(text)):
        if text[j] == "ﬁ" or text[j] == "ﬂ": 
            for a in range(j+1,len(text)):
                offsets[a] = offsets[a] - 1  # Negative offsets for every char after a char replaced with two chars
        elif text[j] in remove_list:
            for a in range(j+1,len(text)):  # Positive offsets for every char after a removed char
                offsets[a] = offsets[a] + 1
        else:
            j += 1
          
    return cleaned_text, offsets

def article_id_to_date(art_id,ca=False):
    ##Convert id to date . CA format  - 0_5_1905-12-28_p1_sn84029386_00295871519_1905122801_0624.json
    
    if  not ca:
        return "-".join(art_id.split("-")[-5:-2])
    else:
        return "-".join(art_id.split("_")[2:3])

def tokenize_text(text):
    
    inputs = TokenizerInfoFast(text) if tokenizer.is_fast else TokenizerInfoSlow(text)
    word_count = inputs.word_count()

    # List to hold the cleaned words and their original positions
    cleaned_words = []
    original_indices = []

    for i in range(word_count):
        start, end = inputs.word_to_chars(i)
        word = text[start:end]
        # Check if the word contains any alphanumeric character
        if any(c.isalnum() for c in word):
            cleaned_words.append(word)
            original_indices.append(start)

    return cleaned_words, original_indices


def clean_mention_text(text,remove_list=["|",":"]):
    text=text.strip()
    ##Remove unwanted characters if not the first char
    for char in remove_list:
        text=text[0]+text[1:].replace(char,"")
    ##Collapse multiple spaces to one
    text=" ".join(text.split())
    return text


def convert_to_disamb_format(article_dict, entity_types_to_keep=["PER"]):
    """The dict contains two lists - ner_words and ner_labels. We need to get it to the format : 'context' - raw text , 'mention_text' - how the entity is mentioned, 'mention_start' - where it starts and 'mention_end'. 
    The output list will have as many elements as there are entity_types_to_keep entity tags. We can assign a new id to each replication - simply add the order of the entity to the id"""
    
    
    ner_words=article_dict["ner_words"]
    ner_tags=article_dict["ner_labels"]
    article=article_dict["article"]
    article_id=article_dict["id"]
    
    
    
    ##Get indices B- tags of only the entity_types_to_keep
    b_tags=[i for i, tag in enumerate(ner_tags) if tag.startswith("B-") and tag.split("-")[1] in entity_types_to_keep]
    
    
    ##Attach continuing I- tags to the B- tags and combine consecutive I- tags
    entity_spans=[]
    for i in b_tags:
        entity_start=i
        entity_end=i
        while entity_end+1<len(ner_tags) and ner_tags[entity_end+1]=="I-"+ner_tags[i].split("-")[1]:
            entity_end+=1
        entity_spans.append((entity_start, entity_end))
    
    ##Get the entity text and the context
    ###Add 'mention_text', 'mention_start', 'mention_end', 'context'. 'context' will be the same for all
    
    output_dict={}
    mention_id=0
    for entity_start, entity_end in entity_spans:
        mention_text=" ".join(ner_words[entity_start:entity_end+1])
        mention_start_word=entity_start
        mention_end_word=entity_end
        context=clean_ocr_text(article)[0]
        ##tokenize context
        context_tokenized, context_offsets=tokenize_text(context)
        
        ##confirm that ner_words==context_tokenized
        assert(ner_words==context_tokenized)
        
        ##Get mention start in original context
        mention_start=context_offsets[mention_start_word]
        mention_end=context_offsets[mention_end_word]+len(ner_words[mention_end_word])
    

            
        ##Clean up the mention_text portion in context. Remove "|", collapse multiple spaces to one, strip leading and trailing spaces
        cleaned_mention_text=clean_mention_text(context[mention_start:mention_end], remove_list=["|",":","\\","&"])
        ##Adjust the context as well and then adjust mention_start and end again
        context=context[:mention_start]+cleaned_mention_text+context[mention_end:]
        mention_end=mention_start+len(cleaned_mention_text)
        
        # print("....")
        # if context[mention_start:mention_end]!=mention_text:
        #     print(ner_words)
        #     print(context)
        #     print(context[mention_start:mention_end],":Tokenized:", mention_text)
        #     input()       
 

        disamb_output_mention={"context":context, "mention_text":cleaned_mention_text, "mention_start":mention_start, "mention_end":mention_end, mention_id:mention_id}
        art_mention_id=str(mention_id)+"_"+article_id
        output_dict[art_mention_id]=disamb_output_mention
        mention_id=mention_id+1

    return  output_dict ##List of dicts


def process_file(file_name):
    # This function will be executed by each worker in the pool.
    # Load the pickle file
    print(f"Processing {file_name}")
    
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        # Make a dict - article: disamb format
        disamb_data = {}
        for article in data.keys():
            disamb_data.update(convert_to_disamb_format(data[article]))

        # Save as the same pickle file name
        output_file = OUTPUT_DIR + "/" + file_name.split("/")[-1]
        with open(output_file, 'wb') as f:
            pickle.dump(disamb_data, f)

        return f"Processed {file_name}"
    except Exception as e:
        return f"Error processing {file_name}: {e}"


def get_fp_embeddings(fp_path, embedding_model_path, special_tokens=None, ent_featurisation='ent_mark',
                      use_multi_gpu=False,pre_featurised=True,
                      re_embed=False,override_max_seq_length=None,asymm_model=False):

    new_embeddings_needed = True
    save_path = f'{embedding_model_path}/fp_embeddings.pkl'

    with open(fp_path) as f:
        fp_raw = json.load(f)
        
    #prototype by keeping only the first 1000
    # fp_raw = {k:fp_raw[k] for k in list(fp_raw.keys())[:10000]}

    # Create list of wikipedia IDs
    dict_list = []
    wik_ids = []
    qid_list = []
    median_year_list=[]
    birth_year_list=[]
    qrank_list=[]

    for _, fp in tqdm(fp_raw.items()):

        if fp != {}:
            wik_ids.append(fp['wiki_title'])
            dict_list.append(fp)
            qid_list.append(fp['wikidata_info']['wikidata_id'])
            median_year_list.append(fp['median_year'])
            birth_year_list.append(fp['birth_year'])
            qrank_list.append(fp['qrank'])

    # Load embeddings if previously created
    if os.path.exists(save_path) and re_embed == False:

        print("Previous embeddings found, loading previous embeddings ...")
        with open(save_path, 'rb') as f:
            embeddings = pickle.load(f)

        if len(embeddings) == len(wik_ids):

            print('Loaded embeddings have same length as data. Using loaded embeddings')
            new_embeddings_needed = False

        else:
            print('Loaded embeddings are different length to data. Creating new embeddings')

    else:
        print("No previous embeddings found./ Rembedding is set to ",re_embed)

    # Otherwise create new embeddings
    if new_embeddings_needed:
        print("Creating new embeddings ...")

        # Load model
        model = SentenceTransformer(embedding_model_path)
        tokenizer = model.tokenizer

        for fp in dict_list:
            fp['context'] = fp['text']

        if not pre_featurised or os.path.exists(f'{embedding_model_path}/featurised_fps.pkl') == False:
            print("Featurising data ...")
            if not asymm_model:
                featurised_fps = featurise_data_with_dates_flex(dict_list,  ent_featurisation, "prepend_1", special_tokens, tokenizer, override_max_seq_length)
            else:
                print("Using asymmetric model featurisation - prepend featurisation and 'FP' key for each fp.")
                featurised_fps = featurise_data(dict_list, featurisation="prepend", special_tokens=special_tokens,  model=SentenceTransformer("all-mpnet-base-v2"),override_max_seq_length=override_max_seq_length)
                ###For each fp, make it a dict instead. {'FP':text}
                featurised_fps = [{'FP':fp} for fp in featurised_fps]
            ##Save featurised data
            with open(f'{embedding_model_path}/featurised_fps.pkl', 'wb') as f:
                pickle.dump(featurised_fps, f)
        else:
            with open(f'{embedding_model_path}/featurised_fps.pkl', 'rb') as f:
                featurised_fps = pickle.load(f)
        
        if use_multi_gpu:
            print("Using multiple GPUs: ", torch.cuda.device_count())
            pool = model.start_multi_process_pool()
            embeddings = model.encode_multi_process(featurised_fps, batch_size=720, pool=pool)
        else:
            embeddings = model.encode(featurised_fps, show_progress_bar=True, batch_size=720)

        # Normalize the embeddings to unit length
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Save embeddings
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
    instance_types_to_keep=set(['human'])

    BIRTH_DATE_CUTOFF=1970     
    QIDS_TO_REMOVE=["Q26702663","Q2292195","Q2600236",
                    "Q112039672","Q5442741","Q16354385",
                    "Q61600540","Q19276771","Q5456005","Q4659108","Q6264176","Q5300213"]
    
    ##Load more QIDs to remove
    with open("qids_to_prune.json") as f:
        more_qids = json.load(f)
        
    ##Add to the list
    QIDS_TO_REMOVE.extend(more_qids)
    QIDS_TO_REMOVE=set(QIDS_TO_REMOVE)
    BIRTH_DATES_ONLY=True
    ##Clean up the dict - to conform with the embeddings - remove empty dicts. 
    fp_dict_cleaned = {k:v for k,v in fp_raw.items() if v!={}}
    
    indices_to_keep = []
    
    ##Find indices that belong to the instance types to keep. It helps that python preserves the order of the dict. entity type is in dict['wikidata_info']['instance_of_labels']
    if len(instance_types_to_keep)>0:
        ##By instance type
        indices_to_keep = [i for i, (k,v) in enumerate(tqdm(fp_dict_cleaned.items())) if len(set(v['wikidata_info']['instance_of_labels']).intersection(instance_types_to_keep))>0]
        ##By date cutoff
        print("indices after instance type filtering: ",len(indices_to_keep))
    if BIRTH_DATE_CUTOFF:
        indices_to_keep_dates = [i for i, (k,v) in enumerate(tqdm(fp_dict_cleaned.items())) if v['birth_year']<BIRTH_DATE_CUTOFF or pd.isna(v['birth_year'])]
        indices_to_keep=set(indices_to_keep).intersection(indices_to_keep_dates)
        print("indices after date filtering: ",len(indices_to_keep))
    
    if QIDS_TO_REMOVE:
        indices_to_remove = [i for i, (k,v) in enumerate(tqdm(fp_dict_cleaned.items())) if v['wikidata_info']['wikidata_id'] in QIDS_TO_REMOVE]
        indices_to_keep=set(indices_to_keep).difference(indices_to_remove)
        print("indices after qid filtering: ",len(indices_to_keep))
    
    if BIRTH_DATES_ONLY:
      ##Drop if birth year or death year is not available
      indices_to_keep_births = [i for i, (k,v) in enumerate(tqdm(fp_dict_cleaned.items())) if not pd.isna(v['birth_year']) or not pd.isna(v['death_year'])]
      indices_to_keep=set(indices_to_keep).intersection(indices_to_keep_births)
      print("indices after birth/death year filtering: ",len(indices_to_keep))
    ##Filter the embeddings and ids
    print("Number of embeddings: ",len(embeddings))
    indices_to_keep=set(indices_to_keep)

    fp_embeddings = np.array([v for i,v in enumerate(tqdm(embeddings)) if i in indices_to_keep])
    fp_ids = [k for i,k in enumerate(wik_ids) if i in indices_to_keep]
    qid_list=[k for i,k in enumerate(qid_list) if i in indices_to_keep]
    median_year_list=[k for i,k in enumerate(median_year_list) if i in indices_to_keep]
    birth_year_list=[k for i,k in enumerate(birth_year_list) if i in indices_to_keep]
    qrank_list=[k for i,k in enumerate(qrank_list) if i in indices_to_keep]
    
    ##Make qrank None to 0
    qrank_list=[0 if pd.isna(q) or q==None else q for q in qrank_list]
    
    print("Number of embeddings after type filtering: ",len(fp_embeddings))
    
    ##Filter the fp_dict_cleaned
    fp_dict_cleaned = {k:v for i,(k,v) in enumerate(tqdm(fp_dict_cleaned.items())) if i in indices_to_keep}
        
    fp_dict_only_text = [v['text'] for k,v in fp_dict_cleaned.items()]
    
    print(len(fp_embeddings), len(fp_ids))
    # assert len(fp_embeddings)==len(fp_ids)==len(fp_dict_only_text) 
    print("All lengths are equal")

    return fp_embeddings, fp_ids, qid_list, median_year_list, birth_year_list, qrank_list, fp_dict_only_text


def rerank_by_qrank(qrank_list, qrank_rerank_threshold, distances, neighbours):
    """Rerank based on QRank. Higher QRANK is better."""
    reranked_indices=[]
    reranked_distances=[]
    for i, nn_list in enumerate(neighbours):
        ##Get the indices of the nearest neighbours
        ##Threshold is determined by nearest neighbour distance - qrank_rerank_threshold
        nearest_nn_distance=distances[i][0]
        ##Threshold = nearest neighbour distance - qrank_rerank_threshold
        threshold=nearest_nn_distance-qrank_rerank_threshold
        distances_i = distances[i]
        nn_indices = [n for j, n in enumerate(nn_list) if distances[i][j] > threshold]
        distances_i = [d for j, d in enumerate(distances_i) if distances[i][j] > threshold]
        ##Get the qrank of the query
        query_qrank = qrank_list[i]
        ##Get the qrank of the nearest neighbours
        nn_qranks = [qrank_list[n] for n in nn_indices]
        
        if len(nn_indices)==0:
            reranked_indices.append(nn_list)
            reranked_distances.append(distances_i)
            continue
        
        ##If all the nearest neighbours have a qrank
        if all([pd.notna(y) for y in nn_qranks]):
            ##Sort by the difference in qrank - max to min
            qrank_diffs = [-abs(query_qrank-y) for y in nn_qranks]
            reranked_indices_i = [n for _, n in sorted(zip(qrank_diffs,nn_indices))]
            reranked_distances_i = [d for _, d in sorted(zip(qrank_diffs,distances_i))]
            reranked_indices.append(reranked_indices_i)
            reranked_distances.append(reranked_distances_i)
            
        else:
            reranked_indices.append(nn_list)
            reranked_distances.append(distances_i)
            
    return reranked_indices, reranked_distances


##STEP1 - Convert the NER outputs to disamb format - article:disamb_list

def format_data():
    INPUT_DIR="/mnt/data01/entity/ner_outputs"
    OUTPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted"
    tokenizer_name="roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    file_list=glob(INPUT_DIR+"/*.pkl")
    print("Files to process:", len(file_list))

    ##Remove files done
    files_done=glob(OUTPUT_DIR+"/*.pkl")
    print("Files done:", len(files_done))
    ##replace output_dir to input dir in files_done
    files_done=[file.replace(OUTPUT_DIR, INPUT_DIR) for file in files_done]

    file_list=[file for file in file_list if file not in files_done]
    print("Files to process:", len(file_list))

    pool = Pool(processes=16)  # None defaults to os.cpu_count()

    # Use pool.map to apply 'process_file' to all elements in 'file_list'
    for result in tqdm(pool.imap(process_file, file_list), total=len(file_list), desc="Files processed"):
        print(result)

    pool.close()
    pool.join()


##STEP 1* - Embed disamb format articles using coref model

def feat_data():
    INPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted"
    OUTPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised"
    files_done=glob(OUTPUT_DIR+"/*.pkl")
    files_done=[file.replace(OUTPUT_DIR, INPUT_DIR) for file in files_done]
    stoks={'men_start': "[M]", 'men_end': "[/M]", 'men_sep': "[MEN]"}
    file_list=glob(INPUT_DIR+"/*.pkl")
    file_list=[file for file in file_list if file not in files_done]
    print("Files to process:", len(file_list))
    for file_name in tqdm(file_list, total=len(file_list), desc="files done"):


        ##Load the pickle file
        with open(file_name, 'rb') as f:
            data = pickle.load(f)


        ##Make a dict - article: disamb format
        article_key_list=list(data.keys())
        print(article_key_list[0])
        article_dict_list=[data[article] for article in article_key_list]
        print(article_dict_list[0])
        disamb_data_feat=featurise_data(article_dict_list, featurisation="ent_mark", special_tokens=stoks,  model=SentenceTransformer("all-mpnet-base-v2"),override_max_seq_length=256)
        data_featurised=dict(zip(article_key_list, disamb_data_feat))
        ##Save as the same pickle file name
        with open(OUTPUT_DIR+"/"+file_name.split("/")[-1], 'wb') as f:
            pickle.dump(data_featurised, f)


##Step 2- Embed disamb format articles using coref model  - save as mention_article_id:embedding

def embed_coref():
    INPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised"
    OUTPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised_corr_embedded"

    files_done=glob(OUTPUT_DIR+"/*.pkl")
    files_done=[file.replace(OUTPUT_DIR, INPUT_DIR) for file in files_done]

    file_list=glob(INPUT_DIR+"/*.pkl")
    file_list=[file for file in file_list if file not in files_done]

    print("Files to process:", len(file_list))
    model_path= "../trained_models/cgis_model_ent_mark_incontext"
    model=SentenceTransformer(model_path)
    for file_name in tqdm(file_list, total=len(file_list), desc="files done"):
        if file_name in files_done:
            continue
        ##Load the pickle file
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        ##Make a dict - article_mention_id: embedding
        ids=list(data.keys())
        texts=list(data.values())
        # pool = model.start_multi_process_pool()
        # embeddings = model.encode_multi_process(texts, batch_size=512, pool=pool)
        embeddings=model.encode(texts, batch_size=720,show_progress_bar=True)

        ##save dict
        output_dict=dict(zip(ids, embeddings))

        ##Save as the same pickle file name
        with open(OUTPUT_DIR+"/"+file_name.split("/")[-1], 'wb') as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


##Step 3 - Cluster emb within date, then, make a dict with date_cluster_id as key and dict of mention_ids as value

def coref_cluster():
    INPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised_corr_embedded"
    OUTPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised_corr_embedded_date_clustered_conservative"
    CLUSTERING="DATE" #Or ARTICLE or DATE
    assert  CLUSTERING in ["ARTICLE", "DATE"]
    file_list=glob(INPUT_DIR+"/*.pkl")
    print("Files to process:", len(file_list))
    files_done=glob(OUTPUT_DIR+"/*.pkl")
    files_done=[file.replace(OUTPUT_DIR, INPUT_DIR) for file in files_done]
    file_list=[file for file in file_list if file not in files_done]
    print("Files to process:", len(file_list))

    for file_name in tqdm(file_list, total=len(file_list), desc="files done"):
        ##Load the pickle file
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        if "_ca" in file_name:
            ca=True
        else:
            ca=False
        ##Make a dict with date:mention_ids
        cluster_dict={}
        for mention_id in data.keys():
            if CLUSTERING=="DATE":
                cluster_unit=article_id_to_date(mention_id,ca)
            else:
                cluster_unit="_".join(mention_id.split("_")[1:])
            if cluster_unit not in cluster_dict:
                cluster_dict[cluster_unit]={}
            cluster_dict[cluster_unit][mention_id]=data[mention_id]


        ###We now want to cluster the embeddings within each date
        unit_cluster_ids={} ##Dict that stores the mention ids and embedding for each cluster in the date.
        for i in tqdm(range(len(cluster_dict.keys()))):
            unit=list(cluster_dict.keys())[i]
            mention_ids=list(cluster_dict[unit].keys())
            mention_embeddings=list(cluster_dict[unit].values())
            mention_embeddings=np.array(mention_embeddings)
            if not mention_embeddings.shape[0]==1:
                # cluster_ids=get_cluster_assignment("agglomerative", cluster_params={'threshold': 0.15, 'clustering linkage': 'average', 'affinity': 'cosine'}, corpus_embeddings=mention_embeddings)
                cluster_ids=get_cluster_assignment("agglomerative", cluster_params={'threshold': 0.10, 'clustering linkage': 'average', 'affinity': 'cosine'}, corpus_embeddings=mention_embeddings)
            else:
                cluster_ids=[0]

            ###Collect the embeddings in each cluster, average them and save as date_cluster_id:{'embedding', 'cluster_mention_ids'}. cluster_mention_ids are the mention_ids in the cluster
            for cluster_id in set(cluster_ids):
                cluster_mention_ids=[mention_ids[i] for i in range(len(mention_ids)) if cluster_ids[i]==cluster_id]
                cluster_embeddings=[mention_embeddings[i] for i in range(len(mention_ids)) if cluster_ids[i]==cluster_id]
                cluster_embedding=np.mean(cluster_embeddings, axis=0)
                date_cluster_id=str(cluster_id)+"_"+unit
                cluster_output={'embedding':cluster_embedding, 'cluster_mention_ids':cluster_mention_ids}
                unit_cluster_ids[date_cluster_id]=cluster_output

        date_entity_count={unit:len(cluster_dict[unit]) for unit in cluster_dict.keys()}
        print(date_entity_count)
        ##Save the dict
        with open(OUTPUT_DIR+"/"+file_name.split("/")[-1], 'wb') as f:
            pickle.dump(unit_cluster_ids, f, protocol=pickle.HIGHEST_PROTOCOL)


##Step 4 - Embed the articles using the disamb model
def embed_disamb():
  INPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised"
  OUTPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised_disamb_embedded"
  
  files_done=glob(OUTPUT_DIR+"/*.pkl")
  files_done=[file.replace(OUTPUT_DIR, INPUT_DIR) for file in files_done]
  
  file_list=glob(INPUT_DIR+"/*.pkl")
  file_list=[file for file in file_list if file not in files_done]
  
  
  model_path= "../trained_models/cgis_model_ent_mark_incontext_disamb_tuned_nodate_shuffled_2e6"
  model=SentenceTransformer(model_path)
  for file_name in tqdm(file_list, total=len(file_list), desc="files done"):
      if file_name in files_done:
          continue
      ##Load the pickle file
      with open(file_name, 'rb') as f:
          data = pickle.load(f)
          
      ##Make a dict - article_mention_id: embedding
      ids=list(data.keys())
      texts=list(data.values())
      # pool = model.start_multi_process_pool()
      embeddings = model.encode(texts, batch_size=720, show_progress_bar=True)
      
      ##save dict
      output_dict=dict(zip(ids, embeddings))
      
      ##Save as the same pickle file name
      with open(OUTPUT_DIR+"/"+file_name.split("/")[-1], 'wb') as f:
          pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


##Step 5: Search for nearest neighbor in wiki corpus and get the QID of the date_cluster_id. Count #mentions

def search_qid():
  INPUT_DIR_EMBEDDINGS="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised_disamb_embedded"
  INPUT_DIR_CLUSTERS="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised_corr_embedded_date_clustered"
  INPUT_DIR_FORMATTED="/mnt/data01/entity/ner_outputs_disamb_formatted"
  OUTPUT_DIR="/mnt/data01/entity/ner_outputs_disamb_formatted_coref_featurised_corr_embedded_date_clustered_wiki_qid"
  
  RERANK_QRANK=True
  
  files_done=glob(OUTPUT_DIR+"/*.pkl")
  files_done=[file.replace(OUTPUT_DIR, INPUT_DIR_CLUSTERS) for file in files_done]
  
  file_list=glob(INPUT_DIR_CLUSTERS+"/*.pkl")
  # file_list=[file for file in file_list if file not in files_done]
  stoks={'men_start': "[M]", 'men_end': "[/M]", 'men_sep': "[MEN]"}
  trained_model_path="../trained_models/cgis_model_ent_mark_incontext_disamb_tuned_nodate_shuffled_2e6"
  ###
  fp_embeddings, fp_ids, qid_list, median_year_list, birth_year_list,qrank_list, fp_dict_only_text = get_fp_embeddings(
    fp_path = '../formatted_first_para_data_qid_template_people_with_qrank_3occupations.json',
    embedding_model_path = trained_model_path,
    special_tokens = stoks,
    use_multi_gpu=True,
    override_max_seq_length=256,
    re_embed=False,
    pre_featurised=True,
    asymm_model=False,        
  )
  
  ###Now search for the nearest neighbor in the fp_embeddings. 
  print("Setting up index")
  res = faiss.StandardGpuResources()
  index = faiss.GpuIndexFlatIP(res, fp_embeddings.shape[1])
  
  ##normalize
  fp_embeddings=fp_embeddings/np.linalg.norm(fp_embeddings, axis=1, keepdims=True)
  index.add(fp_embeddings)

  for file_name in tqdm(file_list, total=len(file_list), desc="files done"):
    ##First, check an example file
    with open(file_name, 'rb') as f:
        cluster_data = pickle.load(f)
    
    with open(file_name.replace(INPUT_DIR_CLUSTERS, INPUT_DIR_FORMATTED), 'rb') as f:
        text_data = pickle.load(f)
    
    with open(file_name.replace(INPUT_DIR_CLUSTERS, INPUT_DIR_EMBEDDINGS), 'rb') as f:
        disamb_emb_data = pickle.load(f)
        
    ##Keep only cluster_mention_ids
    cluster_data={key: cluster_data[key]['cluster_mention_ids'] for key in cluster_data.keys()}
    
    
    ##Now, attach the embeddings to the cluster_mention_ids and then average them to get the cluster embedding
    cluster_data_emb={key: [disamb_emb_data[mention_id] for mention_id in cluster_data[key]] for key in cluster_data.keys()}
    cluster_text_data_mention_text={key: [text_data[mention_id]["mention_text"] for mention_id in cluster_data[key]] for key in cluster_data.keys()}
    # cluster_text_data_context={key: [text_data[mention_id]["context"] for mention_id in cluster_data[key]] for key in cluster_data.keys()}
    ##Average the embeddings
    cluster_data_emb_avg={key: np.mean(cluster_data_emb[key], axis=0) for key in cluster_data_emb.keys()}
    
    
    
    query_embds=np.array(list(cluster_data_emb_avg.values()))
    query_embds=query_embds/np.linalg.norm(query_embds, axis=1, keepdims=True)
    
    print(query_embds.shape)
    ##Search only 100
    if RERANK_QRANK:
      D, I = index.search(query_embds, 10)
      I, D = rerank_by_qrank(qrank_list, 0.01, D, I)
      I=[n[0] for n in I]
      D=[n[0] for n in D]

    else:
      print("Searching")
      D, I = index.search(query_embds, 1)

    ##Get the qid of the nearest neighbor
    qid_list_nn=[qid_list[i] for i in I.flatten()] if not RERANK_QRANK else [qid_list[i] for i in I]
    fp_ids_nn=[fp_ids[i] for i in I.flatten()] if not RERANK_QRANK else [fp_ids[i] for i in I]
      
    
    ##Now, prepare the output dict - also add mention_text and distance
    output_dict={key: {"qid": qid_list_nn[i], 
                      "fp_id": fp_ids_nn[i],
                      "mention_text": cluster_text_data_mention_text[key],
                      # "context_list": cluster_text_data_context[key],
                        "art_mention_ids": cluster_data[key] } for i, key in enumerate(cluster_data_emb_avg.keys())}
    
    ##Add distance
    # output_dict={key: {**output_dict[key], "distance": D[i][0]} for i, key in enumerate(output_dict.keys())} if not RERANK_QRANK else {key: {**output_dict[key], "distance": D[i]} for i, key in enumerate(output_dict.keys())}
    
    ##make pandas df
    output_df=pd.DataFrame(output_dict).T
    output_df["date_cluster_id"]=output_df.index
    
    ##Add distance
    output_df["distance"]=D.flatten() if not RERANK_QRANK else D
    
    ##Sort by distance
    output_df=output_df.sort_values("distance", ascending=False)
    
    ###Write
    output_df.to_csv(OUTPUT_DIR+"/"+file_name.split("/")[-1].replace(".pkl", ".csv"), index=False)
    
    
    
    #Save as json
    with open(OUTPUT_DIR+"/"+file_name.split("/")[-1], 'w') as f:
        json.dump(output_dict, f)


##Run as script
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run entity disambiguation pipeline')
    parser.add_argument('--step', type=str, help='Step to run', required=True)
    args = parser.parse_args()

    if args.step=="format_data":
        format_data()
    elif args.step=="featurise_data":
        feat_data()
    elif args.step=="embed_coref":
        embed_coref()
    elif args.step=="coref_cluster":
        coref_cluster()
    elif args.step=="embed_disamb":
        embed_disamb()
    elif args.step=="search_qid":
        # coref_cluster()
        search_qid()