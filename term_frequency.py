import sys
import os
from nltk.corpus import stopwords

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_DIR = os.fsencode(os.path.join(CUR_DIRECTORY, "Test Files"))

stop_words = set(stopwords.words("english"))

##########################################
#             FILE READER                #
##########################################

def read_files():
    for file in os.listdir(TEST_FILE_DIR):
        filename = os.fsencode(file)
        if filename.endswith(".txt"):
            with open(filename, 'r') as text_file:
                data = text_file.read().replace('\n', '')


def remove_stopwords(data):
    # Remove stop words
    text_to_parse = [word for word in data if word not in stop_words]


##########################################
#              MAIN METHOD               #
##########################################

if __name__ == '__main__':
    read_files()

