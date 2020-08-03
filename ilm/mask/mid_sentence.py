from enum import Enum

import random
from random import randrange

from ..string_util import doc_to_hierarchical_offsets

from .base import MaskFn


class MaskMSentenceType(Enum):
    SENTENCE = 0


class MaskMSentence(MaskFn):
    def __init__(self, p=1.0 / 3.0, verse=False):
        if not verse:
            from nltk.tokenize import sent_tokenize
            try:
                sent_tokenize('Ensure punkt installed.')
            except:
                raise ValueError('Need to call nltk.download(\'punkt\')')
        self.p = p
        self.verse = verse

    @classmethod
    def mask_types(cls):
        return list(MaskMSentenceType)

    @classmethod
    def mask_type_serialize(cls, m_type):
        return m_type.name.lower()

    def mask(
            self,
            doc,
            mask_sentence_p=None):
        doc_offs = doc_to_hierarchical_offsets(doc, verse=self.verse)

        mask_sentence_p = self.p if mask_sentence_p is None else mask_sentence_p

        def _trial(p):
            if p <= 0:
                return False
            else:
                return random.random() < p

        masked_spans = []

        doc_off, doc_len, p_offs = doc_offs
        for p_off, p_len, s_offs in p_offs:
            if len(s_offs) > 3:
                # mask every middle sentence by p
                # for s_ind, (s_off, s_len, w_offs) in enumerate(s_offs[1:-1]):
                #     if _trial(mask_sentence_p):
                #         masked_spans.append((MaskMSentenceType.SENTENCE, s_off, s_len))
                # random mask one sentence from middle sentences
                k = randrange(len(s_offs)-2) + 1
                s_off, s_len, _ = s_offs[k]
                masked_spans.append((MaskMSentenceType.SENTENCE, s_off, s_len))


        return masked_spans


class MaskMSentenceVerse(MaskMSentence):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, verse=True, **kwargs)
