maxSeqLength = 250
batchSize = 24

import numpy as np
import tensorflow as tf
import re

wordsList = np.load('other_data/wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('other_data/wordVectors.npy')
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999
    return sentenceMatrix

inputText = "That movie was terrible."
inputMatrix = getSentenceMatrix(inputText)
print inputMatrix
print inputMatrix.shape
np.savetxt("inputMatrixNegative.csv", inputMatrix, delimiter=',', fmt="%i")

secondInputText = "That movie was the best one I have ever seen."
secondInputMatrix = getSentenceMatrix(secondInputText)
print secondInputMatrix
print secondInputMatrix.shape
np.savetxt("inputMatrixPositive.csv", secondInputMatrix, delimiter=',', fmt="%i")
