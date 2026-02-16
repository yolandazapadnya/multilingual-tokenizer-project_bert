# Multilingual Tokenizer & Sentence Embedding Analysis

This project performs a basic multilingual text analysis combining token-based statistics with neural sentence embeddings.

It is designed as a small, reproducible NLP pipeline suitable for linguistic analysis and as a portfolio project for NLP/AI roles. It includes six Indo-European languages: European Portuguese, Ukrainian, Russian, English, Spanish, and German: 50 sentences in total.

## What the script does

The script `text_tokenizer.py`:

1. Loads a text file (`data.txt`) containing numbered sentences
2. Cleans and extracts the sentence text
3. Computes token-level statistics using a multilingual BERT tokenizer
4. Computes sentence embeddings using a multilingual SentenceTransformer model
5. Stores summary results in a CSV file

## Models used

- **Tokenizer**: `bert-base-multilingual-cased` (for token counts only)
- **Sentence embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`
  (SentenceTransformers, PyTorch backend)

This setup avoids unstable low-level forward passes on macOS while retaining neural representations.

## How to run

From the project directory:

```bash
source venv/bin/activate
python text_tokenizer.py

The results.csv file is merely an example of a previously run model and it does not reflect the results for the current code.
