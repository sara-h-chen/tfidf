import os
import string
import numpy as np
import knapsack
import unittest
import operator

from spacy.lang.en.stop_words import STOP_WORDS

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_DIR = os.path.join(CUR_DIRECTORY, "Test Files")

# Remove contractions
STOP_WORDS.add('s')
STOP_WORDS.add('ll')
STOP_WORDS.add('t')

dictionary_of_words = {}
common_word = {}
score_list = []


##########################################
#             TEST CASES                 #
##########################################

class TermFrequencyTest(unittest.TestCase):
    def test(self):
        test_dict = {}
        tracked = {}
        count_words(test_dict, tracked, 0, 'the quick brown fox jumps over the lazy dog'.split(" "))
        calculate_tf(test_dict, tracked, 0)

        self.assertEqual(test_dict['quick'][0]['total'], 1)
        self.assertEqual(test_dict['quick'][0]['tf'], 0.75)
        self.assertEqual(tracked[0]['word'], 'the')
        self.assertEqual(tracked[0]['count'], 2)


##########################################
#             FILE READER                #
##########################################

def read_files(text):
    # Replace new lines
    lines = text.read().replace('\n', ' ')
    data = lines.lower()

    # Remove all numbers
    numset = '0123456789'
    removed_num = data.translate(str.maketrans(numset, ' ' * len(numset)))

    # Remove all punctuation
    # Initialize punctuation remover
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    removed_punc = removed_num.translate(translator)
    processed, length = remove_stopwords(removed_punc)
    return lines, processed, length


##########################################
#           TEXT MANIPULATOR             #
##########################################

def remove_stopwords(data):
    # Remove stop words
    data = data.split(" ")
    spaces_removed = list(filter(None, data))

    length = len(spaces_removed)
    text_to_parse = [word for word in spaces_removed if word not in STOP_WORDS]
    return text_to_parse, length


##########################################
#                 COUNT                  #
##########################################

def count_words(word_dict, tracked_common_word, doc_id, text):
    for word in text:
        if word in word_dict:
            if doc_id in word_dict[word]:
                word_dict[word][doc_id]['total'] += 1
            else:
                word_dict[word].update({doc_id: {'total': 1}})
        else:
            word_dict.update({word: {doc_id: {'total': 1}}})

        # Track the most frequent word
        if doc_id in tracked_common_word:
            if word_dict[word][doc_id]['total'] > tracked_common_word[doc_id]['count']:
                tracked_common_word[doc_id]['count'] = word_dict[word][doc_id]['total']
                tracked_common_word[doc_id]['word'] = word
        else:
            tracked_common_word.update({doc_id: {'word': word, 'count': word_dict[word][doc_id]['total']}})

    return word_dict


# Augmented frequency
# 0.5 + ( 0.5 * raw count / (max raw frequency of the most occurring term in the docs) )
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
def calculate_tf(word_dict, most_common, doc_id):
    for key, value in word_dict.items():
        # if the word is in document we're currently looking at
        if doc_id in word_dict[key]:
            term_frequency = 0.5 + (0.5 * (word_dict[key][doc_id]['total'] / most_common[doc_id]['count']))
            word_dict[key][doc_id]['tf'] = term_frequency
    return


def calculate_average_tf(chunk_number, doc_id, sentences):
    for j in range(0, len(sentences)):
        sentence_score = 0
        sentence = sentences[j].split(" ")
        for word in sentence:
            if word in dictionary_of_words:
                sentence_score += dictionary_of_words[word][doc_id]['tf']
        average_tf = sentence_score / len(sentence)

        # If less than half a Tweet long, begin penalizing the score
        if len(sentences[j]) < 70:
            average_tf *= (len(sentences[j]) / 100)

        mult = (3 - chunk_number)
        score_list.append({'index': j, 'average_tf': average_tf, 'length': len(sentence),
                           'chunk': chunk_number, 'doc_id': doc_id,
                           'multiplier': (((mult - 1) * 0.25) + 1.0) * average_tf})
    return score_list


##########################################
#          STRUCTURAL CONTEXT            #
##########################################

# UNUSED
def split_chunks(long_list, no_words):
    split = np.array_split(long_list, 3)
    for j in range(0, len(split)):
        no_words += (j * len(split[j]))
    return split


##########################################
#            UNPACK SUMMARY              #
##########################################

def reconstruct(corpus, chosen_sentences):
    # Sort them by the order in which they appear
    chosen_sentences.sort(key=operator.itemgetter('doc_id', 'chunk', 'index'))
    print(chosen_sentences)
    output_summary = ""
    for sentence in chosen_sentences:
        sentence_index = sentence['index']
        chunk_index = sentence['chunk']
        document_no = sentence['doc_id']
        output_summary += corpus[document_no][chunk_index][sentence_index]
        output_summary += ". "
    return output_summary


##########################################
#              MAIN METHOD               #
##########################################

if __name__ == '__main__':
    lmt = 250
    use_tfidf = False

    complete_corpus = []

    # RUN UNIT TEST
    # unittest.main()

    # Loop through all given documents
    document_number = 0
    for file in os.listdir(TEST_FILE_DIR):
        filename = os.path.join(TEST_FILE_DIR, os.fsencode(file).decode())
        if filename.endswith('.txt'):
            with open(filename, 'r') as text_file:
                # Preprocess the files, calculating the tf value for each word
                doc_string, for_preprocessing, doc_length = read_files(text_file)
                count_words(dictionary_of_words, common_word, document_number, for_preprocessing)
                calculate_tf(dictionary_of_words, common_word, document_number)

                # DEBUG
                # print(dictionary_of_words)
                # print(common_word)

                # Use structural information
                # Split the document into 3 different sections; give each different weights
                sentence_chunks = np.array_split(list(filter(None, doc_string.split(". "))), 3)
                complete_corpus.append(sentence_chunks)

                # Calculate the average tf for each sentence
                for i in range(0, len(sentence_chunks)):
                    calculate_average_tf(i, document_number, sentence_chunks[i])

                # Normalize by document length
                for score in score_list:
                    # noinspection PyTypeChecker
                    score['normalized_tf'] = score['multiplier'] / doc_length

            document_number += 1

    # DEBUG
    # print(score_list)
    # print(dictionary_of_words)

    # Use dynamic programming to find the best sentences to include
    bagged = knapsack.knapsack01_dp(score_list, lmt, use_tfidf)
    val, wt = knapsack.total_value(bagged, lmt)
    print("Reconstructed summary for a total value of %f and a total weight of %i" % (val, -wt))

    summary = reconstruct(complete_corpus, bagged)
    print("Summary ------>> ")
    print(summary)
