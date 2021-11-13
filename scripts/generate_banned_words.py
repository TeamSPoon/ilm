import sys

import nltk


banned = {"CC", "DT",  "EX", "IN", "MD", "POS", "PRP$", "RP", "TO", "WDT", "WP"}

#with open("./resources/common_words.txt") as f:
#    filler_words = set([l.lower()[:-1] for l in f.readlines()])
filler_words = set()    
with open(sys.argv[1]) as f:
    lines = f.readlines()


# tokene into words
tokens = nltk.word_tokenize("\n".join(lines))

# parts of speech tagging
tagged = nltk.pos_tag(tokens)

filler_words = filler_words |  set([w.lower() for w,pos in tagged if pos in banned and w.isalpha()])

with open(sys.argv[2], 'w') as f:
    f.write("\n".join(sorted(list(filler_words))))
