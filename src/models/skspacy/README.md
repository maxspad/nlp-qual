# skspacy
Wrapping Spacy in Scikit-Learn Transformers

## Prerequisites
- `python=3.8.12`
- `numpy=1.21.3`
- `scikit-learn=1.0`
- `spacy=3.1.3`
- `tqdm=4.62.3`

## Usage
The library provides three classes `SpacyTransformer`, `SpacyTokenFilter` and `SpacyDocFeats`:
- `SpacyTransformer` - this takes an (n, 1) NumPy ndarray of text documents and uses Spacy to process them, yielding a flat list of processed Spacy Documents. This is the starting point.
- `SpacyTokenFilter` - Given a (n, 1) ndarray of Spacy Docs, filters the tokens within, optionally lemmatizing, for downstream processing by scikit-learn's `CountVectorizer` or `TfidfVectorizer`
- `SpacyDocFeats` - Generate whole-document features, such as document vectors, part of speech counts, token counts, and entity type counts from spacy Documents. 

The classes are designed to be compatible with scikit-learn's `Transformers`/`Pipeline` architecture.
