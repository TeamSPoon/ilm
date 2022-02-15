import random

sentences = []
with open(f"/u/scr/mhahn/software/sent-conv-torch/data/rt-polarity.all", "r", encoding='latin-1') as inFile:
  for line in inFile:
     line = line.strip()
     line = line[line.index(" ")+1:]
     sentences.append(line)
with open(f"MR/train.txt", "w") as outFile:
   print('\n\n\n'.join(sentences), file=outFile)
