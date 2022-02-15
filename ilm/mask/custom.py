from enum import Enum
import random

from .base import MaskFn



class MaskTokenNumberType(Enum):
  ONE = 0
  TWO = 1
  THREE = 2
  FOUR = 3
  FIVE = 4
  SIX = 5
  SEVEN_OR_MORE = 6



class MaskTokenNumber(MaskFn):
  def __init__(self, p=0.15):
    self.p = p

  @classmethod
  def mask_types(cls):
    return list(MaskTokenNumberType)

  @classmethod
  def mask_type_serialize(cls, m_type):
    return m_type.name.lower()

  def mask(self, doc):
    #print(doc)
    split = doc.split(" ")
    assert split[1] == "@", split
    i = 2
    masked_spans = []
    while i < len(split):
      if random.random() < self.p:
         # mask starting at i
         choice = random.random()
         if choice < 0.4 or i+1 == len(split):
             length = 1
             span_type = MaskTokenNumberType.ONE
         elif choice < 0.6 or i+2 == len(split):
             length = 2
             span_type = MaskTokenNumberType.TWO
         elif choice < 0.65 or i+3 == len(split):
             length = 3
             span_type = MaskTokenNumberType.THREE
         elif choice < 0.75 or i+4 == len(split):
             length = 4
             span_type = MaskTokenNumberType.FOUR
         elif choice < 0.8 or i+5 == len(split):
             length = 5
             span_type = MaskTokenNumberType.FIVE
         elif choice < 0.85 or i+6 == len(split):
             length = 6
             span_type = MaskTokenNumberType.SIX
         elif choice < 0.9 or i+7 == len(split):
             length = 7
             span_type = MaskTokenNumberType.SEVEN_OR_MORE
         elif choice < 0.95 or i+8 == len(split):
             length = 8
             span_type = MaskTokenNumberType.SEVEN_OR_MORE
         elif choice < 1.0:
             length = 9
             span_type = MaskTokenNumberType.SEVEN_OR_MORE
         words = split[i:i+length]
 #        print(words, length)
         masked_spans.append((span_type, i+sum([len(x) for x in split[:i]]), length -1 + sum([len(x) for x in split[i:i+length]])))
#         print([doc[masked_spans[-1][1]:masked_spans[-1][1]+masked_spans[-1][2]]])
         i += length
      else:
         i += 1
#    for span_offset, char in enumerate(doc):
#      if not char.isalnum() and len(char.strip()) > 0 and random.random() < self.p:
#        if char in ['.', '?', '!']:
#          span_type = MaskTokenNumberType.SENTENCE_TERMINAL
#        else:
#          span_type = MaskTokenNumberType.OTHER
#        span_len = 1
#        masked_spans.append((span_type, span_offset, span_len))
    return masked_spans




class MaskPunctuationType(Enum):
  SENTENCE_TERMINAL = 0
  OTHER = 1


class MaskPunctuation(MaskFn):
  def __init__(self, p=0.5):
    self.p = p

  @classmethod
  def mask_types(cls):
    return list(MaskPunctuationType)

  @classmethod
  def mask_type_serialize(cls, m_type):
    return m_type.name.lower()

  def mask(self, doc):
    masked_spans = []
    for span_offset, char in enumerate(doc):
      if not char.isalnum() and len(char.strip()) > 0 and random.random() < self.p:
        if char in ['.', '?', '!']:
          span_type = MaskPunctuationType.SENTENCE_TERMINAL
        else:
          span_type = MaskPunctuationType.OTHER
        span_len = 1
        masked_spans.append((span_type, span_offset, span_len))
    return masked_spans


from nltk import pos_tag
from ..string_util import word_tokenize
from ..tokenize_util import tokens_offsets

class MaskProperNounType(Enum):
  PROPER_NOUN = 0

class MaskProperNoun(MaskFn):
  def __init__(self, p=1.):
    try:
      pos_tag(['Ensure', 'tagger'])
    except:
      raise ValueError('Need to call nltk.download(\'averaged_perceptron_tagger\')')
    self.p = p

  @classmethod
  def mask_types(cls):
    return list(MaskProperNounType)

  @classmethod
  def mask_type_serialize(cls, m_type):
    return m_type.name.lower()

  def mask(self, doc):
    from nltk import pos_tag
    masked_spans = []
    toks = word_tokenize(doc)
    toks_offsets = tokens_offsets(doc, toks)
    toks_pos = pos_tag(toks)
    for t, off, (_, pos) in zip(toks, toks_offsets, toks_pos):
      if pos == 'NNP' and random.random() < self.p:
        masked_spans.append((MaskProperNounType.PROPER_NOUN, off, len(t)))
    return masked_spans

class MaskFillerWordType(Enum):
  FILLER_WORD = 0
  FILLER_NGRAM = 1

class MaskFillerWords(MaskFn):
  def __init__(self, p=1.):
      self.p = .8
      with open("scripts/resources/filler_words.txt") as f:
          self.filler_words = set([l[:-1] for l in f.readlines()])
  @classmethod
  def mask_types(cls):
    return list(MaskFillerWordType)

  @classmethod
  def mask_type_serialize(cls, m_type):
    return m_type.name.lower()

  def mask(self, doc):
    masked_spans = []
    toks = word_tokenize(doc)
    toks_offsets = tokens_offsets(doc, toks)
    last_word_ngram = False
    for t, off in zip(toks, toks_offsets):
      if t.lower() in self.filler_words and random.random() < self.p:
        if last_word_ngram:
          prev = masked_spans.pop()
          masked_spans.append((MaskFillerWordType.FILLER_NGRAM, prev[1], (off-prev[1])+len(t)))
        else:
          masked_spans.append((MaskFillerWordType.FILLER_WORD, off, len(t)))
        last_word_ngram = True
      else:
        last_word_ngram = False
    return masked_spans
    
