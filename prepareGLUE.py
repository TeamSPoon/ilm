import glob
import random


tasks = ["CoLA", "MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B", "WSC"]

partition = "dev"

sentences = []

for task in tasks:
  if task == "MNLI" and partition == "dev":
   inputs = [open(x, "r") for x in sorted(glob.glob(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{task}/processed/{partition}_matched.raw.input*"))]
  elif task != "WSC":
   inputs = [open(x, "r") for x in sorted(glob.glob(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{task}/processed/{partition}.raw.input*"))]
  else:
   inputs = [open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/WNLI/{partition}.tsv", "r")]
  data = [x.read().strip().split("\n") for x in inputs]
  print([len(x) for x in data])
  _ = (x.close() for x in inputs)
  for i in range(len(data[0])):
     if task != "WSC":
       sent = task + " @ " + " ".join([x[i] for x in data])
     else:
       if i == 0:
          continue
       else:
           sent = task + " @ " + data[0][i].split("\t")[1]
     sentences.append(sent)
with open(f"GLUE/{partition}.txt", "w") as outFile:
   print('\n\n\n'.join(sentences), file=outFile)
