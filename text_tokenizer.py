#1. Open the dataset
with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
#2. Remove numbering of the sentences
clean_data = []

for line in lines:
    # remove newline
    line = line.strip()
    
    # split only at the first ". "
    sentence = line.split(". ", 1)
    
    clean_data.append(sentence)

print(clean_data)

#3. Run the tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

for sentence in clean_data:
    tokens = tokenizer.tokenize(sentence)
    print(tokens)

#4. Store results in pandas (optional, but better if I want to have the results stored)
import pandas as pd

results = []

for sentence in clean_data:
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    
    results.append({
    "sentence": sentence,
    "num_tokens": len(tokens),
    "num_characters": len(sentence),
    "token_char_ratio": len(tokens) / len(sentence)
})

df = pd.DataFrame(results)
print(df)

print("Average tokens per sentence:", df["num_tokens"].mean())
print("Max tokens:", df["num_tokens"].max())
print("Min tokens:", df["num_tokens"].min())

print("\nSummary statistics:")
print(f"Average tokens per sentence: {df['num_tokens'].mean():.2f}")
print(f"Maximum tokens in a sentence: {df['num_tokens'].max()}")
print(f"Minimum tokens in a sentence: {df['num_tokens'].min()}")

#5. Save results to csv
df.to_csv("results.csv", index=False)

print("Analysis complete. Results saved to results.csv")
