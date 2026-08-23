README

Description

This is a small semantic search engine. It takes a set of text documents, converts each one into a numeric vector using a sentence embedding model, and lets you search them with a plain text query. Instead of matching exact words, it matches meaning, so a query can find a document even if they do not share the same words.

Setup

1. Make sure Python 3.8 or newer is installed.
2. Install the required package.

pip install sentence-transformers --break-system-packages

3. Run the script.

python search_engine.py

The first run downloads a small embedding model from the internet, about 80 megabytes. After that it is cached locally and no internet connection is needed.

Working

The script keeps a dictionary of documents.

Each document is converted into an embedding vector using the encode function. This vector is a list of numbers that represents the meaning of the sentence.

All document vectors are stored in an index, built once when the script starts.

When you type a search term, it is also converted into an embedding vector using the same encode function.

The script then compares the query vector against every document vector using cosine similarity. This produces a score between 0 and 1, where a higher score means the document is closer in meaning to the query.

Results are sorted from highest score to lowest and printed to the screen, along with a short preview of each matching document.

If you want to skip embeddings and use the original word counting method instead, set use_embeddings to False when creating the VectorCompare object. The rest of the script will keep working without changes.
