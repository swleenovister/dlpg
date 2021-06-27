__all__ = ['WhiteSpaceTokenizer']

from konlpy.tag import Komoran
import constant

komoran = Komoran()

def WhiteSpaceTokenizer(document):
    return document.split()

def KomoranMorphemAnalyzer(document):
    global komoran
    return ['/'.join(i) for i in komoran.pos(document)]

class ToPaddedTokenSequence:
    def __init__(self, vocab):

class ToPaddedTokenSequence:
    def __init__(self, vocab):

class Vocabulary:
    def __init__(self):
        pass

    @staticmethod
    def build_vocab(vocab):
        if type(vocab) == str:
            vocab = Vocabulary.load_vocab