# Multilingual Tokenizer Analysis

This project analyzes tokenization behavior of a multilingual BERT model in six Indo-European languages: European Portuguese, Ukrainian, Russian, English, Spanish, and German.

## Goal

To examine how subword tokenization behaves across sentences in different languages.

## Method

1. Since this is an example project, the sentences were generated with the help of Claude Sonnet 4.5
2. Then, the sentences were loaded into a numbered text file (.txt, created in VS Code) and placed inside the folder
3. After that, the data file was loaded
4. Next, sentence numbering was removed
5. The model `bert-base-multilingual-cased` was used
6. Number of tokens per sentence was calculated

## Output

The script generates `results.csv` containing sentence-level token counts.

## How to run

```bash
python text_tokenizer.py
