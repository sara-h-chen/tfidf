import os
import string
import numpy as np

from spacy.lang.en.stop_words import STOP_WORDS

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_DIR = os.path.join(CUR_DIRECTORY, "Test Files")

# Remove contractions
STOP_WORDS.add('s')
STOP_WORDS.add('ll')
STOP_WORDS.add('t')

dictionary_of_words = {}
most_frequent_word = {'word': '', 'count': 0}

# Augmented frequency
# 0.5 + ( 0.5 * raw count / (max raw frequency of the most occurring term in the docs) )
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf


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
# 0 - Located early in the article, given more weight
def count_words(doc, list_of_chunks):
    length = len(list_of_chunks)
    for i in range(0, length):
        for word in list_of_chunks[i]:
            if word in dictionary_of_words:
                # Track the document number
                if doc in dictionary_of_words[word]:
                    # Track the section in which the word lies
                    # 0 - Start of doc, 1 - middle of doc, 2 - end of doc
                    if i in dictionary_of_words[word][doc]:
                        dictionary_of_words[word][doc][i] += 1
                    else:
                        dictionary_of_words[word][doc].update({i: 1})
                else:
                    dictionary_of_words[word].update({doc: {i: 1}})

                # Give a weighted score based on location in document
                dictionary_of_words[word][doc]['score'] += ((length - i) * 1)
                ttl = dictionary_of_words[word][doc]['total'] + 1
                dictionary_of_words[word][doc]['total'] = ttl

                # Track most frequent word
                if ttl > most_frequent_word['count']:
                    most_frequent_word['word'] = word
                    most_frequent_word['count'] = ttl
            else:
                dictionary_of_words.update({word: {doc: {i: 1, 'score': ((length - i) * 1), 'total': 1}}})
    return dictionary_of_words


##########################################
#                 COUNT                  #
##########################################

def counter():
    # for key, value in dictionary_of_words.items():
    #
    return


##########################################
#          STRUCTURAL CONTEXT            #
##########################################

# Split the document into different sizes; give each different weights
def split_chunks(long_list, no_words):
    split = np.array_split(long_list, 3)
    for i in range(0, len(split)):
        no_words += (i * len(split[i]))
    return split


##########################################
#              MAIN METHOD               #
##########################################

if __name__ == '__main__':
    number_of_words = 0

    # Initialize punctuation remover
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # Loop through all given documents
    document_number = 0
    for file in os.listdir(TEST_FILE_DIR):
        filename = os.path.join(TEST_FILE_DIR, os.fsencode(file).decode())
        if filename.endswith('.txt'):
            with open(filename, 'r') as text_file:
                for_parsing = read_files(text_file)
                sorted_chunks = count_words(document_number, split_chunks(for_parsing, number_of_words))
                # DEBUG
                print(sorted_chunks)
                print(most_frequent_word)
                document_number += 1
                exit()
