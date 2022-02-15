import random


for partition in ["train", "dev", "test"]:
 sentences = []
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/{partition}.tsv", "r") as inFile:
  header = next(inFile) # for the header
  header = header.strip().split("\t") # "sentence\tlabel\n", header
  sent_col = header.index("sentence")
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 2:
       continue
     assert len(line) == 2
     sentences.append(line[sent_col])
 with open(f"SST2/{partition if partition != 'dev' else 'valid'}.txt", "w") as outFile:
   print('\n\n\n'.join(sentences), file=outFile)
