MODEL_DIR = "train_SST2/"
MASK_CLS = 'ilm.mask.hierarchical.MaskHierarchical'


import os
import pickle

import ilm.tokenize_util

tokenizer = ilm.tokenize_util.Tokenizer.GPT2
with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
    additional_ids_to_tokens = pickle.load(f)
additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
try:
    ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
except ValueError:
    print('Already updated')
print(additional_tokens_to_ids)

# Load model

import torch
from transformers import GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()
_ = model.to(device)


context = """
.  _ funny _ bad behavior .
""".strip()

context_ids = ilm.tokenize_util.encode(context, tokenizer)
print(context_ids)
# Replace blanks with appropriate tokens from left to right
_blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']
context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']
#context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_ngram|>']
#context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']
#context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']
#context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']
print(ilm.tokenize_util.decode(context_ids, tokenizer))


from ilm.infer import infill_with_ilm

generated = infill_with_ilm(
    model,
    additional_tokens_to_ids,
    context_ids,
    num_infills=10)
for g in generated:
    print('-' * 80)
    print(ilm.tokenize_util.decode(g, tokenizer))


blankCandidates = []

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_c_sentBreak_new_finetuned_large.tsv", "r") as inFile:
   for line in inFile:
       if line.startswith("####"):
          next(inFile)
          tokenized = next(inFile).strip().split(" ")
          print("TOK", tokenized)
          line = next(inFile)
       if len(line) < 3:
        continue
       try:
          mask, sampled = line.strip().split("\t")
       except ValueError:
          continue
       sampled = sampled.strip().split(" ")
       mask = mask.strip()
       assert len(sampled) == len(mask), (sampled, mask)
       masked = [sampled[i] if mask[i] == "0" else "[MASK]" for i in range(len(mask))]
#       print(mask)
 #      print(masked)
 #      print(tokenized)
       masked = "".join(masked).replace("▁", " ").replace("[MASK]", " _ ").replace("  ", " ").replace("</s>", "").strip()
  #     print(masked)
       blankCandidates.append((" ".join(tokenized), mask, masked))
   #    if len(blankCandidates) > 1000:
  #       quit()

queue = []
processed = set()
with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_ILM.tsv", "w") as outFile:
  for tokenized, mask, masked in blankCandidates:
       if (tokenized, mask, masked) in processed:
         continue
       processed.add((tokenized, mask, masked))
 #      print(masked)
       context = masked
       if context[0] == "_":
          context = " "+context
       _blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
       
       context_ids = ilm.tokenize_util.encode(context, tokenizer)
#       print(context_ids)
       i = 0
       while i < len(context_ids):
          if context_ids[i] == _blank_id:
            print(i)
            for j in range(i, len(context_ids)):
               if context_ids[j] != _blank_id:
                    break
            #print(j)
            if j - i < 2:
               context_ids[i] = additional_tokens_to_ids['<|infill_word|>']
            else:
               context_ids[i] = additional_tokens_to_ids['<|infill_ngram|>']
            context_ids = context_ids[:i+1] + context_ids[j:]
          i+=1 
       print(ilm.tokenize_util.decode(context_ids, tokenizer))
       
       
       from ilm.infer import infill_with_ilm
       
       generated = infill_with_ilm(
           model,
           additional_tokens_to_ids,
           context_ids,
           num_infills=10)
       for g in generated:
           decoded = ilm.tokenize_util.decode(g, tokenizer)
           print(mask, decoded)
           print(mask, "\t", tokenized, "\t", decoded, file=outFile)
       
       
