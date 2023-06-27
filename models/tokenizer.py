from abc import ABC, abstractmethod

import underthesea
import nltk

nltk.download('punkt')


class BaseTokenizer(ABC):

    @abstractmethod
    def tokenize(self):
        pass


class ViTokenizer(BaseTokenizer):

    def tokenize(self, sentence):
        return underthesea.word_tokenize(sentence)


class EnTokenizer(BaseTokenizer):

    def tokenize(self, sentence):
        return nltk.tokenize.word_tokenize(sentence)