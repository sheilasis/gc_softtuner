import prompt_tuner

import os
import termcolor
import numpy as np
import sys

#1729
universe = sys.argv[1]
#EleutherAI/gpt-neo-1.3B
modelo =  sys.argv[2]
#80 ["20", "40", "60", "80"]
soft_dim = int(sys.argv[3])
#For 13B adjust the batch size to 400, everything else can remain 2048
bsize = int(sys.argv[4])
#@markdown For most use cases 1 epoch is enough, increase if you use a very small dataset.
nepoch = int(sys.argv[5])
#16 - If you picked a small batch size increase this value accordingly (Suggested values are in the dropdown) ["16", "64"]
nsteps = int(sys.argv[6])
#3e-5 
lrate = float(sys.argv[7])


trainer = prompt_tuner.BasicTrainer(universe, quiet=True)
print(termcolor.colored("\n\nDone.\n\n", "green"), flush=True)
trainer.data.ckpt_path = modelo
trainer.get_hf_checkpoint_metadata()
trainer.data.save_file = "softprompt.mtjsp"
trainingprompt = ""

if not trainingprompt:
  trainer.data.prompt_method = "vocab_sample"
  #In case you left the prompt blank you can specify here how many tokens your prompt takes up (Larger is more knowledge but takes up more space from the story context when you use your softprompt)
  trainer.data.soft_in_dim = soft_dim
else:
  with open(trainingprompt) as f:
	  initial_softprompt = f.read()
  tokenizer = trainer.get_tokenizer()
  if trainer.data.newlinemode == "s":  # Handle fairseq-style newlines if required
      initial_softprompt = initial_softprompt.replace("\n", "</s>")
  trainer.data.initial_softprompt = tokenizer.encode(
      initial_softprompt, max_length=int(2e9), truncation=True
  )

dataset_path = "dataset/" 
output_file = "dataset.npy"
batch_size = bsize
epochs = nepoch

print(termcolor.colored("\n\nTokenizing Dataset.\n\n", "green"), flush=True)
trainer.tokenize_dataset(dataset_path, output_file, batch_size, epochs)
print(termcolor.colored("\n\nStarting Training.\n\n", "green"), flush=True)

dataset_file = output_file
trainer.data.dataset_file = dataset_file
trainer.data.gradient_accumulation_steps = nsteps


learning_rate = lrate 
trainer.data.stparams = {
    "lr": learning_rate,
    "max_grad_norm": 10.0,
    "weight_decay": 0.1,
    "warmup": 0.1,
    "end_lr_multiplier": 0.1,
    "save_every": 10,
}

# Now, begin training!
trainer.train()

print(termcolor.colored("\n\nSave Kobold.\n\n", "green"), flush=True)

output_file = "sheila_prompt.zip"
name = "sheila_prompt"
supported = trainer.data.ckpt_path
description = "softprompt test"
author = "Sheila"
trainer.export_to_kobold(output_file, name, author, supported, description)

output_file_name = "sheila_prompt.json"
soft_prompt_name = name
soft_prompt_description = supported + " - " + description
trainer.export_to_mkultra(output_file_name, soft_prompt_name, soft_prompt_description)
