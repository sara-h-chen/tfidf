import os
import string
import numpy as np
import knapsack
import unittest
import operator

from spacy.lang.en.stop_words import STOP_WORDS

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# WARNING: Only put the docs you want to summarize in this directory
TEST_FILE_DIR = os.path.join(CUR_DIRECTORY, "Test Files")

# Remove contractions
STOP_WORDS.add('s')
STOP_WORDS.add('ll')
STOP_WORDS.add('t')

dictionary_of_words = {}
document_lengths = {}
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


def calculate_average_tf(chunk_number, doc_id, sentences, document_length):
    for j in range(0, len(sentences)):
        sentence_score = 0
        sentence = sentences[j].split(" ")

        # If it is not an empty sentence
        for word in sentence:
            if word in dictionary_of_words:
                sentence_score += dictionary_of_words[word][doc_id]['tf']

        average_tf = sentence_score / len(sentence)

        # If less than half a Tweet long, begin penalizing the score
        if len(sentences[j]) < 70:
            average_tf *= (len(sentences[j]) / 100)

        # Set up a multiplier that tells you which 1/3 the sentence lies in
        mult = (3 - chunk_number)
        # Apply multiplier; is later normalized by the length of the document
        after_multiplier = (((mult - 1) * 0.25) + 1.0) * average_tf
        score_list.append({'index': j, 'average_tf': average_tf,
                           'length': len(sentence), 'chunk': chunk_number,
                           'doc_id': doc_id, 'multiplier': after_multiplier,
                           'normalized_tf': (after_multiplier / document_length)})
    return score_list


def calculate_idf(dictionary, doc_count):
    for word, value in dictionary.items():
        number_of_appearances = len(dictionary[word].keys())
        # Adjusted for division by 0
        idf = np.log(doc_count / (1 + number_of_appearances))
        dictionary[word]['idf'] = idf
    return


def calculate_tfidf(dictionary, doc_id, section, sentence_index, corpus, scores, score_list_index):
    specific_sentence = corpus[doc_id][section][sentence_index]
    split_sentence = specific_sentence.split(" ")
    total_tfidf = 0
    for word in split_sentence:
        if word in dictionary:
            total_tfidf += (dictionary[word][doc_id]['tf'] * dictionary[word]['idf'])
    averaged_tfidf = total_tfidf / len(split_sentence)

    # If less than half a Tweet long, begin penalizing the score
    if len(specific_sentence) < 70:
        averaged_tfidf *= (len(specific_sentence) / 100)

    scores[score_list_index]['average_tfidf'] = averaged_tfidf

    # Apply multiplier; is later normalized by the length of the document
    mult = (3 - section)
    after_multiplier = (((mult - 1) * 0.25) + 1.0) * averaged_tfidf
    scores[score_list_index]['multiplier_tfidf'] = after_multiplier
    scores[score_list_index]['normalized_tfidf'] = (after_multiplier / document_lengths[doc_id])
    return


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
    use_tfidf = True

    complete_corpus = []

    # RUN UNIT TEST
    # unittest.main()

    # Loop through all given documents
    document_number = 0
    # NOTE: Only the files that are to be summarized are put in this folder
    corpus_files = os.listdir(TEST_FILE_DIR)
    number_of_docs = len(corpus_files)
    for file in corpus_files:
        filename = os.path.join(TEST_FILE_DIR, os.fsencode(file).decode())
        if filename.endswith('.txt'):
            with open(filename, 'r') as text_file:
                # Pre-process the files, calculating the tf value for each word
                doc_string, for_preprocessing, doc_length = read_files(text_file)
                count_words(dictionary_of_words, common_word, document_number, for_preprocessing)
                calculate_tf(dictionary_of_words, common_word, document_number)

                # DEBUG
                # print(dictionary_of_words)
                # print(common_word)

                # Use structural information
                # Split the document into 3 different sections; give each different weights
                document_lengths.update({document_number: doc_length})
                sentence_chunks = np.array_split(list(filter(None, doc_string.split(". "))), 3)
                complete_corpus.append(sentence_chunks)

                # Calculate the average tf for each sentence
                for i in range(0, len(sentence_chunks)):
                    calculate_average_tf(i, document_number, sentence_chunks[i], doc_length)

            document_number += 1

    # DEBUG
    # print(score_list)
    # print(dictionary_of_words)

    if use_tfidf:
        # Calculate tf-idf
        # complete_corpus[i][j][k]
        # i = doc_id
        # j = section in document
        # k = sentence in chunk
        sentence_counter = 0
        calculate_idf(dictionary_of_words, number_of_docs)
        for doc in range(0, len(complete_corpus)):
            for chunk in range(0, len(complete_corpus[doc])):
                for sentence in range(0, len(complete_corpus[doc][chunk])):
                    # DEBUG
                    # print(complete_corpus[doc][chunk][sentence])
                    # print(score_list[sentence_counter])

                    calculate_tfidf(dictionary_of_words, doc, chunk, sentence,
                                    complete_corpus, score_list, sentence_counter)
                    sentence_counter += 1

    # Use dynamic programming to find the best sentences to include
    bagged = knapsack.knapsack01_dp(score_list, lmt, use_tfidf)
    val, wt = knapsack.total_value(bagged, lmt)
    print("Reconstructed summary for a total value of %f and a total weight of %i" % (val, -wt))

    summary = reconstruct(complete_corpus, bagged)
    print("Summary ------>> ")
    print(summary)
