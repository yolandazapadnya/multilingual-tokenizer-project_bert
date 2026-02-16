import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
print("SentenceTransformer model loaded OK")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=False)

#1. Open the dataset
with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

#2. Remove numbering of the sentences
clean_data = []

for line in lines:
    line = line.strip()
    if not line:
        continue

    parts = line.split(". ", 1)

    if len(parts) == 2:
        sentence = parts[1]
    else:
        sentence = parts[0]

    clean_data.append(sentence)

#3. Store results in pandas (optional, but better if I want to have the results stored)

results = []

for sentence in clean_data[:50]:
    # symbolic analysis
    tokens = tokenizer.tokenize(sentence)

    # sentence embedding (SentenceTransformer API)
    embedding = model.encode(sentence)

    results.append({
        "sentence": sentence,
        "num_tokens": len(tokens),
        "num_characters": len(sentence),
        "token_char_ratio": len(tokens) / len(sentence),
        "embedding_norm": float(np.linalg.norm(embedding)),
        "embedding_dim": int(embedding.shape[0])
    })

df = pd.DataFrame(results)

print("\nSummary statistics:")
print(f"Average tokens per sentence: {df['num_tokens'].mean():.2f}")
print(f"Maximum tokens in a sentence: {df['num_tokens'].max()}")
print(f"Minimum tokens in a sentence: {df['num_tokens'].min()}")

#4. Save results to csv
df.to_csv("results.csv", index=False)

print("Analysis complete. Results saved to results.csv")
