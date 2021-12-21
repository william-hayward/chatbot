import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')  # downloads the latest version of 'punkt'
porterstemmer = PorterStemmer()  # defines the stemming protocol used in the program


def tokenize(sentence):
    return nltk.word_tokenize(sentence)  # tokenizes a sentence that is passed through the function and returns it


def stem(words):
    return porterstemmer.stem(words.lower())  # function stems a list of words that are passed through and returns it


def bag_of_words(tokenized_sentence, words):
    sentence = [stem(i) for i in tokenized_sentence]  # stems each word in the sentence
    bag = np.zeros(len(words), dtype=np.float32)  # starts off as setting the value for each word in the bag as zero
    for i, j in enumerate(words):
        if j in sentence:
            bag[i] = 1.0  # if the words in the bag are the same as words in the tokenized sentence the value is set
            # to one
    return bag
