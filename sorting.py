import pickle
from ilm.mask.hierarchical import MaskHierarchicalType
import sys

def load_pickle(pickle_filename, number_of_documents):
    arxiv_masks = pickle.load(open(pickle_filename, "rb"))
    if number_of_documents != 0:
        return arxiv_masks[:number_of_documents]
    else:
        return arxiv_masks

def count_mask_types(masking):
    count = [0, 0, 0, 0, 0]
    for mask in masking:
        count[mask[0].value] += 1

    return count

def filter_words(array, ordered_masks):
    # ONLY WORD maskings
    for document in array:
        bool_arr = [count_mask_types(masking)[4] == len(masking) for masking in document[1]]
        word_maskings = [document[1][idx] for idx, e in enumerate(bool_arr) if e == True]
        ordered_masks.append((document[0], word_maskings))

def filter_ngrams(array, ordered_masks):
    # ONLY WORD OR NGRAM maskings
    for document in array:
        bool_arr = []
        for masking in document[1]:
            c = count_mask_types(masking)
            bool_arr.append(c[3] != 0 and c[4] + c[3] == len(masking)) 
        ngram_maskings = [document[1][idx] for idx, e in enumerate(bool_arr) if e == True]
        ordered_masks.append((document[0], ngram_maskings))

def filter_sentences(array, ordered_masks):
    # ONLY WORD OR NGRAM OR SENTENCE maskings
    for document in array:
        bool_arr = []
        for masking in document[1]:
            c = count_mask_types(masking)
            bool_arr.append(c[2] != 0 and c[4] + c[3] + c[2] == len(masking)) 
        sentence_maskings = [document[1][idx] for idx, e in enumerate(bool_arr) if e == True]
        ordered_masks.append((document[0], sentence_maskings))

def filter_paragraphs(array, ordered_masks):
    # ONLY WORD OR NGRAM OR SENTENCE OR PARAGRAPH maskings
    for document in array:
        bool_arr = []
        for masking in document[1]:
            c = count_mask_types(masking)
            bool_arr.append(c[1] != 0 and c[4] + c[3] + c[2] + c[1] == len(masking)) 
        paragraph_maskings = [document[1][idx] for idx, e in enumerate(bool_arr) if e == True]
        ordered_masks.append((document[0], paragraph_maskings))

def filter_documents(array, ordered_masks):
    # if any DOCUMENT maskings present
    for document in array:
        bool_arr = []
        for masking in document[1]:
            c = count_mask_types(masking)
            bool_arr.append(c[0] != 0) 
        document_maskings = [document[1][idx] for idx, e in enumerate(bool_arr) if e == True]
        ordered_masks.append((document[0], document_maskings))

def sort_masking(array):
    ordered_masks = []
    filter_words(array, ordered_masks) 
    filter_ngrams(array, ordered_masks)
    filter_sentences(array, ordered_masks)
    filter_paragraphs(array, ordered_masks)
    return ordered_masks

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Use the script with 3 arguments.")
        print("1. the input pickle filename")
        print("2. the output pickle filename")
        print("3. the number of documents in the input pickle to sort into the output. 20 => first 20 will be sorted and dumped into the output pickle. Use 0 to imply all documents")
    pickle_filename = sys.argv[1]
    number_of_documents = int(sys.argv[3])
        
    result_pickle_filename = sys.argv[2]
    if pickle_filename == result_pickle_filename:
        print("Cannot overwrite input pickle file (argument 1). Use a different output filename (argument 2)")
    else:
        masks = load_pickle(pickle_filename, number_of_documents)
        ordered_masks = sort_masking(masks)
        with open(result_pickle_filename, 'wb') as result_file:
            pickle.dump(ordered_masks, result_file)
