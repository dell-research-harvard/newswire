import string

import json
from statistics import mean, mode, median
from nltk.util import ngrams
import csv
import pandas as pd
import re
import time

import transformers
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers import pipeline
from tqdm import tqdm
import georeferencing_utils as geo_utils
import hashlib
import copy

import byline_detection
import columnist_detection


# City dictionary (key: lowercase name, value: standard name)
with open('geo_files/city_dict.json') as f:
  city_dict = json.load(f)

# City hash table (key: h11 hash value, value: list of all matching cities)
with open('geo_files/hashed_cities.json') as f:
  hashed_cities = json.load(f)

# State dictionary (key: state abb, value: state name)
with open('geo_files/state_dict.json') as f:
  state_dict = json.load(f)

# Country dictionary (key: lowercase name, value: standard name)
with open('geo_files/country_dict.json') as f:
  country_dict = json.load(f)


def get_article_list(clusters):
    art_list = []
    art_ids = []
    count = 0

    for cluster in clusters.values():
        for article_id in cluster:
            art_list.append(cluster[article_id])
            art_ids.append(article_id)

    return art_list, art_ids

def clean_bylines(bylines):
    clean_bylines = {}
    art_ids = []
    byline_list = []

    for art_id, art_bylines in bylines.items():
        if art_bylines:
            for byline in art_bylines:
                art_ids.append(art_id)
                byline_list.append(byline)
        else:
            clean_bylines[art_id] = None

    cleaned_bylines = columnist_detection.remove_columnists(byline_list)

    for idx, byline in enumerate(cleaned_bylines):
        art_id = art_ids[idx]
        if art_id not in clean_bylines:
            clean_bylines[art_id] = []
        clean_bylines[art_id].append(byline)

    return clean_bylines

def get_bylines(clusters, byline_model, tokenizer):
    byline_dict = {}
    
    art_list, art_ids = get_article_list(clusters)
    bylines = byline_detection.batched_bylines(art_list, byline_model, tokenizer, batch_size=512)

    for idx, art_id in enumerate(art_ids):
        byline_dict[art_id] = bylines[idx]

    return clean_bylines(byline_dict)


def get_matches(clusters, art_bylines):
    matches = {}

    for cluster_key, cluster in tqdm(clusters.items()):

        state_suggestions = []
        country_suggestions = []
        city_suggestions = []

        for art_id in list(cluster.keys()):
            city_poss = []
            bylines = art_bylines[art_id]

            if bylines:
                bylines = bylines.copy()   

                all_grams = []
                caps_grams = []

                for byline in bylines:

                    cleaned_bylines = geo_utils.clean(byline).lower()
                    split_bylines = cleaned_bylines.split()

                    only_caps = re.findall('[A-Z]+[A-Z]+[A-Z]*[.!?\\-]*[\s]+', (geo_utils.clean(byline) + " "))
                    only_caps = "".join(only_caps).strip().lower()
                    only_caps = only_caps.split()

                    # Get all ngrams
                    for i in range(1, len(split_bylines) + 1):
                        all_grams.extend([" ".join(list(b)) for b in list(ngrams(split_bylines, i))])

                    for i in range(1, len(only_caps) + 1):
                        caps_grams.extend([" ".join(list(b)) for b in list(ngrams(only_caps, i))])

                all_grams = set(all_grams)
                no_punc = [''.join(c for c in s if c not in string.punctuation) for s in all_grams]
                remove_punc = [geo_utils.remove_punct(s) for s in all_grams if len(s) > 3]
                all_grams.update(set(no_punc))
                all_grams.update(set(remove_punc))

                caps_grams = set(caps_grams)
                no_punc = [''.join(c for c in s if c not in string.punctuation) for s in caps_grams]
                remove_punc = [geo_utils.remove_punct(s) for s in caps_grams if len(s) > 3]
                caps_grams.update(set(no_punc))
                caps_grams.update(set(remove_punc))

                for ngram in all_grams:
                    # Match to a country
                    if ngram in country_dict:
                        country_suggestions.append(country_dict[ngram])
                    # Match to a state
                    if ngram in state_dict:
                        state_suggestions.append(state_dict[ngram])

                # Match to a city
                for ngram in caps_grams:
                    if ngram in city_dict:
                        city_poss.append(city_dict[ngram])
                        city_suggestions.append(city_dict[ngram])

                # look at lowercase if we still haven't found city
                if city_poss == []:
                    for ngram in all_grams:
                        if ngram in city_dict and ngram not in country_dict:
                            if ngram in ["washington", "new york", "nevada"] or ngram not in state_dict:
                                city_poss.append(city_dict[ngram])
                                city_suggestions.append(city_dict[ngram])


        matches[cluster_key] = {
            'city_possibilities': list(set(city_suggestions)),
            'city_pick': geo_utils.most_common(city_suggestions, 0.15, len(cluster), True),
            'state_possibilities': list(set(state_suggestions)),
            'state_pick': geo_utils.most_common(state_suggestions, 0.15, len(cluster)),
            'country_possibilities': list(set(country_suggestions)),
            'country_pick': geo_utils.most_common(country_suggestions, 0.15, len(cluster)),
        }

    return matches


def clean_matches(matches):
    # Clean up some matches based on hard rules

    # stand-alone city exceptions
    us_except = {"Atlanta" : "Georgia", "Baltimore" : "Maryland", "Boston" : "Massachusetts", "Chicago" : "Illinois",
                "Cincinnati" : "Ohio", "Cleveland" : "Ohio", "Dallas" : "Texas", "Denver" : "Colorado", "Detroit" : "Michigan",
                "Honolulu" : "Hawaii", "Houston" : "Texas", "Indianapolis" : "Indiana", "Las Vegas" : "Nevada",
                "Los Angeles" : "California", "Miami" : "Florida", "Hollywood" : "California", "Milwaukee" : "Wisconsin",
                "Minneapolis" : "Minnesota", "New Orleans" : "Louisiana", "New York" : "New York", "Oklahoma City" : "Oklahoma",
                "Philadelphia" : "Pennsylvania", "Phoenix" : "Arizona", "Pittsburgh" : "Pennsylvania", "Saint Louis" : "Missouri",
                "Salt Lake City" : "Utah", "San Antonio" : "Texas", "San Diego" : "California",
                "San Francisco" : "California", "Seattle" : "Washington", "Washington" : "District of Columbia"}

    intl_except = {"Beijing" : "China", "Berlin" : "Germany", "Djibouti" : "Djibouti", "Geneva" : "Switzerland", "Gibraltar" : "Gibraltar",
                    "Guatemala City" : "Guatemala", "Havana" : "Cuba", "Hong Kong" : "Hong Kong", "Jerusalem" : "Israel",
                    "Kuwait" : "Kuwait", "London" : "United Kingdom", "Luxembourg" : "Luxembourg", "Macau" : "Macau",
                    "Mexico City" : "Mexico", "Monaco" : "Monaco", "Montreal" : "Canda", "Moscow" : "Russia", "Ottawa" : "Canada",
                    "Paris" : "France", "Quebec" : "Canada", "Rome" : "Italy", "San Marino" : "San Marino", "Singapore" : "Singapore",
                    "Tokyo" : "Japan", "Toronto" : "Toronto", "Vatican City" : "Vatican City", "Madrid" : "Spain"}

    for cluster in matches:

        # if no city was found, get rid of state pick
        if matches[cluster]['city_pick'] == []:
            matches[cluster]['state_pick'] = []

        # if city name is short and no other info is found, assume OCR error
        try:
            if len(matches[cluster]['city_pick'][0]) <= 3 and matches[cluster]['state_pick'] == [] and matches[cluster]['country_pick'] == []:
                matches[cluster]['city_pick'] = []
        except:
            continue

        # if state in US, drop non-US countries
        if matches[cluster]['state_pick'] != [] and matches[cluster]['country_pick'] not in ['United States', 'Canada']:
            matches[cluster]['country_pick'] = []

        # address exceptions
        try:
            if matches[cluster]['city_pick'][0] in us_except and matches[cluster]['state_pick'] == []:
                matches[cluster]['state_pick'] = [us_except[matches[cluster]['city_pick'][0]]]
        except: pass

        # address DC case
        try:
            if matches[cluster]['city_pick'][0] == "Washington" and (matches[cluster]['state_pick'] == [] or matches[cluster]['state_pick'][0] == "Washington"):
                matches[cluster]['state_pick'] = ["District of Columbia"]
        except: pass

        try:
            if matches[cluster]['city_pick'][0] in intl_except and matches[cluster]['country_pick'] == [] and matches[cluster]['state_pick'] == []:
                matches[cluster]['country_pick'] = [intl_except[matches[cluster]['city_pick'][0]]]
        except: pass

        # get rid of cities that are state names
        try:
            if matches[cluster]['city_pick'][0].lower() in state_dict.keys() and matches[cluster]['city_pick'][0].lower() not in ["washington", "new york", "nevada"]:
                if matches[cluster]['city_pick'][0] == matches[cluster]['state_pick'][0]:
                    matches[cluster]['city_pick'] = []
        except: pass

        # change UK countries to UK
        try:
            if matches[cluster]['country_pick'][0] in ["Northern Ireland", "Wales", "Scotland", "England"]:
                matches[cluster]['country_pick'] = ["United Kingdom"]
        except: pass

    with open('geo_files/state_lookup.json') as f:
            state_to_news =  json.load(f)

    for cluster in matches:
        try:
            newspapers = list(matches[cluster]['articles'].keys())
            newspapers = ["-".join(x.split('-')[1:-5]) for x in newspapers]
            states = [state_to_news[x] for x in newspapers]
            state, count = geo_utils.most_frequent(states)
            state = state.replace("-", " ")
            if count > len(newspapers) * 0.9:
                matches[cluster]['state_guess'] = state
        except: pass

    return matches

def get_coords(matches):
    # Define helper functions

    # create dictionaries to convert from states to abbreviations and vice-versa
    with open('geo_files/state_to_abb.json') as f:
        us_state_to_abbrev = json.load(f)

    abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))

    for state in us_state_to_abbrev.copy():
        us_state_to_abbrev[state.lower()] = us_state_to_abbrev[state]

    # define h11 hash function for use in hash table
    def h11(w):
        return hashlib.md5(w).hexdigest()[:9]

    # Harmonize across city, state, country
    us_states=["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

    finalized = copy.deepcopy(matches)

    for cluster in list(finalized.keys()):

        # manually modify some problematic examples
        try:
            if finalized[cluster]['city_pick'][0].lower() == "kansas city":
                finalized[cluster]['state_pick'] = ["Missouri"]
            elif finalized[cluster]['city_pick'][0].lower() == "saint louis":
                finalized[cluster]['city_pick'] = ["St. Louis"]
        except: pass

        state_match = []

        # add in US and Canada country picks if we found a state in the text and we don't already have a country label
        try:
            if finalized[cluster]['state_pick'][0] in us_states and finalized[cluster]['country_pick'] == []:
                finalized[cluster]['country_pick'] = ["United States"]
                state_match = finalized[cluster]['state_pick']
            elif finalized[cluster]['state_pick'][0] not in us_states and finalized[cluster]['country_pick'] == []:
                finalized[cluster]['country_pick'] = ["Canada"]
        except: pass

        city_matches = []
        city_match = finalized[cluster]['city_pick']
        country_match = finalized[cluster]['country_pick']
        state_guess = []

        # add in state guesses
        try:
            if finalized[cluster]['state_pick'] == [] and (finalized[cluster]['country_pick'] == [] or finalized[cluster]['country_pick'] == "United States"):
                state_guess = [finalized[cluster]['state_guess']]
        except: pass

        try:
            name = city_match[0].lower()
        except: pass

        try:
            if state_match != [] or state_guess != []:
                try:
                    abb = us_state_to_abbrev[state_match[0]]
                except:
                    abb = us_state_to_abbrev[state_guess[0]]
                city_matches = sorted([city for city in hashed_cities[h11(name.encode())] if (city['name'].lower() == city_match[0].lower() or city_match[0].lower() in city['alternate_names']) and city['admin1_code'].lower() == abb.lower()], key = lambda x : x['population'], reverse = True)
                # try iowa if louisiana doesn't work
                if city_matches == [] and abb == "LA":
                    abb = "IA"
                    city_matches = sorted([city for city in hashed_cities[h11(name.encode())] if (city['name'].lower() == city_match[0].lower() or city_match[0].lower() in city['alternate_names']) and city['admin1_code'].lower() == abb.lower()], key = lambda x : x['population'], reverse = True)
        except: pass

        # if we still have no matches, restrict by country
        try:
            if state_match == [] and country_match != [] and city_matches == [] and country_match[0]:
                city_matches = sorted([city for city in hashed_cities[h11(name.encode())] if (city['name'].lower() == city_match[0].lower() or city_match[0].lower() in city['alternate_names']) and city['country'].lower() == country_match[0].lower()], key = lambda x : x['population'], reverse = True)
        except: pass

        # if we still have no matches, don't restrict by country
        try:
            if state_match == [] and country_match == [] and city_matches == []:
                city_matches = sorted([city for city in hashed_cities[h11(name.encode())] if (city['name'].lower() == city_match[0].lower() or city_match[0].lower() in city['alternate_names'])], key = lambda x : x['population'], reverse = True)
        except: pass

        # update country pick and coordinates accordingly
        try:
            if city_matches:
                finalized[cluster]['country_pick'] = city_matches[0]['country']
                finalized[cluster]['coords'] = city_matches[0]['coordinates']
                if finalized[cluster]['country_pick'] == "United States":
                    finalized[cluster]['state_pick'] = abbrev_to_us_state[city_matches[0]['admin1_code']]
        except: pass

        # if we don't have coordinates, set picks to empty
        try:
            if 'coords' not in finalized[cluster]:
                finalized[cluster]['city_pick'] = []
                finalized[cluster]['state_pick'] = []
                finalized[cluster]['country_pick'] = []
        except: pass

    capitalized_us = {}

    for state in us_states:
        capitalized_us[state.lower()] = state

    for cluster in finalized:
        if finalized[cluster]['state_pick'] != []:
            if type(finalized[cluster]['state_pick']) is list:
                finalized[cluster]['state_pick'] = finalized[cluster]['state_pick'][0]
                state_pick = finalized[cluster]['state_pick']
                if state_pick in capitalized_us:
                    finalized[cluster]['state_pick'] = capitalized_us[state_pick]

        del finalized[cluster]['city_possibilities']
        del finalized[cluster]['state_possibilities']
        del finalized[cluster]['country_possibilities']

    return finalized


if __name__ == "__main__":

    start = time.time()

    # Load byline detection model
    byline_model = transformers.AutoModelForTokenClassification.from_pretrained('byline_model')
    byline_tokenizer = transformers.AutoTokenizer.from_pretrained('byline_model')

    with open('clusters.json') as f:
        clusters = json.load(f)

    bylines = get_bylines(clusters, byline_model, byline_tokenizer)

    matches = get_matches(clusters, bylines)
    cleaned_matches = clean_matches(matches)
    final_matches = get_coords(cleaned_matches)

    with open('final_matches.json', 'w') as f:
        json.dump(final_matches, f)

    end = time.time()
    print(end - start)
