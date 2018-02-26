import os
import string
import numpy as np

from operator import itemgetter
from spacy.lang.en.stop_words import STOP_WORDS

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_DIR = os.path.join(CUR_DIRECTORY, "Test Files")

# Remove contractions
STOP_WORDS.add('s')
STOP_WORDS.add('ll')
STOP_WORDS.add('t')

dictionary_of_words = {}


##########################################
#             FILE READER                #
##########################################

def read_files(text):
    # Replace new lines
    data = text.read().lower().replace('\n', ' ')
    # Remove all punctuation
    data = data.translate(translator)
    return remove_stopwords(data)


##########################################
#           TEXT MANIPULATOR             #
##########################################

def remove_stopwords(data):
    # Remove stop words
    data = data.split(" ")
    text_to_parse = list(filter(None, [word for word in data if word not in STOP_WORDS]))
    return text_to_parse


# NOTE: Using structural information
# 0 - Located early in the article
def count_words(list_of_chunks):
    for i in range(0, len(list_of_chunks)):
        # DEBUG
        # print(list_of_chunks[i])
        for word in list_of_chunks[i]:
            if word in dictionary_of_words:
                if i in dictionary_of_words[word]:
                    dictionary_of_words[word][i] += 1
                else:
                    dictionary_of_words[word].update({i: 1})
            else:
                dictionary_of_words.update({word: {i: 1}})
    return dictionary_of_words


##########################################
#          STRUCTURAL CONTEXT            #
##########################################

# Return the first 1/3 of the document so that you can give it more weight
def split_chunks(long_list):
    return np.array_split(long_list, 3)


##########################################
#              MAIN METHOD               #
##########################################

if __name__ == '__main__':
    # Initialize punctuation remover
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # Loop through all given documents
    for file in os.listdir(TEST_FILE_DIR):
        filename = os.path.join(TEST_FILE_DIR, os.fsencode(file).decode())
        if filename.endswith('.txt'):
            with open(filename, 'r') as text_file:
                for_parsing = read_files(text_file)
                sorted_chunks = count_words(split_chunks(for_parsing))
                print(sorted_chunks)
                exit()
