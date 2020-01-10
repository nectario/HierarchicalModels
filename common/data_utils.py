from collections import defaultdict

from nltk.corpus import words
import pandas as pd
#from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import os

#from symspellpy.symspellpy import SymSpell
from spacy.lang.en import English
import spacy
import numpy as np
from keras_preprocessing.sequence import pad_sequences

def convert_to_string(token):
    if type(token) == str:
        return token
    return str(token.text)

def check_spelling(words_in_text, spellchecker):
    word_list = []
    misspelled_words = spellchecker.unknown(words_in_text)
    for word in words_in_text:
        if word.lower() in misspelled_words:
            word_list.append(word)
    return word_list

def is_all_lowercase(value):
    lowercase = [c for c in value if c.islower()]
    if len(value) == len(lowercase):
        return True
    else:
        return False

def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError: #not an iterable
        pass


def get_max_shape(array):
    dimensions = defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]

def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError:  # final level
        yield (*index, slice(len(array))), array


def pad_nested_sequences(array, fill_value=0.0):
    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return np.array(result)
