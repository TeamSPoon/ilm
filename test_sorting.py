import unittest
import pickle
from sorting import load_pickle, filter_words, filter_ngrams, filter_sentences, filter_paragraphs, filter_documents, count_mask_types
from ilm.mask.hierarchical import MaskHierarchicalType

class TestSorting(unittest.TestCase):
    def test_pickle_loading(self):
        masks = load_pickle('sample_pickle.pkl', 20) 
        self.assertTrue(len(masks) == 20)

    def test_count_mask_types(self):
        masks = load_pickle('sample_pickle.pkl', 20) 
        self.assertEqual(count_mask_types(masks[0][1][14]), [0, 1, 1, 2, 2])

    def test_filtering_words(self):
        masks = load_pickle('sample_pickle.pkl', 20) 
        result_array = []
        filter_words(masks, result_array)
        for document in result_array:
            for masking in document[1]:
                for mask in masking:
                    self.assertTrue(mask[0] == MaskHierarchicalType.WORD)

    def test_filtering_ngrams(self):
        masks = load_pickle('sample_pickle.pkl', 20) 
        result_array = []
        filter_ngrams(masks, result_array)
        for document in result_array:
            for masking in document[1]:
                for mask in masking:
                    self.assertTrue(mask[0] == MaskHierarchicalType.WORD or mask[0] == MaskHierarchicalType.NGRAM)
    
    def test_filtering_sentences(self):
        masks = load_pickle('sample_pickle.pkl', 20) 
        result_array = []
        filter_sentences(masks, result_array)
        for document in result_array:
            for masking in document[1]:
                for mask in masking:
                    self.assertTrue(mask[0] == MaskHierarchicalType.WORD or mask[0] == MaskHierarchicalType.NGRAM or mask[0] == MaskHierarchicalType.SENTENCE)
    
    def test_filtering_paragraphs(self):
        masks = load_pickle('sample_pickle.pkl', 20) 
        result_array = []
        filter_paragraphs(masks, result_array)
        for document in result_array:
            for masking in document[1]:
                for mask in masking:
                    self.assertTrue(mask[0] == MaskHierarchicalType.WORD or mask[0] == MaskHierarchicalType.NGRAM or mask[0] == MaskHierarchicalType.SENTENCE or mask[0] == MaskHierarchicalType.PARAGRAPH)

    def test_number_of_maskings_remain_same(self):
        number_of_documents = 20
        masks = load_pickle('sample_pickle.pkl', number_of_documents) 
        
        word_masks = []
        ngram_masks = []
        sentence_masks = []
        paragraph_masks = []
        document_masks = []
        filter_words(masks, word_masks)
        filter_ngrams(masks, ngram_masks)
        filter_sentences(masks, sentence_masks)
        filter_paragraphs(masks, paragraph_masks)
        filter_documents(masks, document_masks)
        for didx in range(number_of_documents):
            self.assertEqual(len(word_masks[didx][1]) + len(ngram_masks[didx][1]) + len(sentence_masks[didx][1]) + len(paragraph_masks[didx][1]) + len(document_masks[didx][1]), len(masks[didx][1])) 

if __name__ == '__main__':
    unittest.main()
