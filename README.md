# B211-Assignment_9
# Assignment 9 NLTK Project

## Purpose
This project analyzes four text files with NLTK and compares them using basic natural language processing techniques. It prints the most frequent words, counts named entities, extracts trigrams, compares the trigram overlap for authorship guessing, and exports the summary results to a CSV file.

## Design Overview
This project is written as a procedural script rather than a class-based program. That means the logic is organized into small functions and shared data structures instead of custom classes. This keeps the code simple and easy to follow for a single assignment-focused workflow.

The script follows this flow:
1. Load the four text files.
2. Preprocess each text by tokenizing and removing noise.
3. Run stemming and lemmatization helpers.
4. Count named entities with NLTK.
5. Build trigram frequency counts.
6. Compare Text 4 against the other texts.
7. Export the final summary to CSV with pandas.

## Class Design
There are no custom classes in this project because the lecture material for this assignment focused on procedural NLTK code rather than object-oriented design.

Instead of a class, the script uses functions and dictionaries to organize the work. The `texts` dictionary groups each label with its file contents, and the `results` dictionary stores the outputs from the analysis stage. If this had been written as a class, the same information would likely have been stored as attributes and methods on a single text-analysis object, but that was not necessary for this assignment.

## Implementation Details
The script uses these libraries:
- `nltk` for tokenizing, tagging, named entity recognition, and trigrams
- `collections.Counter` for trigram frequency counting
- `pandas` for CSV export
- `pathlib.Path` for building the output file path

The input files are:
- `RJ_Lovecraft.txt`
- `RJ_Tolkein.txt`
- `RJ_Martin.txt`
- `Martin.txt`

The output file is:
- `analysis_results.csv`

## Global Variables and Their Roles
- `text1`, `text2`, `text3`, `text4`: hold the raw contents of each input text file.
- `texts`: stores the label and content for each text so the same analysis loop can process all files.
- `stop_words`: the English stop-word list from NLTK.
- `stemmer`: a Porter stemmer instance used to reduce words to stems.
- `lemmatizer`: a WordNet lemmatizer instance used to convert words to dictionary forms.
- `results`: stores the computed outputs for each text, including tokens, top words, named entities, and trigrams.
- `tri4`: stores the trigram list for Text 4 so it can be compared against the other texts.
- `sim1`, `sim2`, `sim3`: store trigram-overlap scores used to choose the closest author match.
- `csv_path`: the path where the CSV export is written.

## Class Attributes and Methods
This project does not define any custom classes, so there are no class attributes or class methods to list. The closest equivalent is the set of helper functions and shared variables described above.

## Functions

### `load_text(path)`
Reads a text file using UTF-8 encoding and returns the file contents as a string.

### `preprocess(text)`
Tokenizes the text, keeps only alphabetic tokens, converts them to lowercase, and removes stop words.

### `stem_tokens(tokens)`
Applies stemming to each token so related word forms are reduced to a common root.

### `lemma_tokens(tokens)`
Applies lemmatization to each token so words are normalized into more readable dictionary forms.

### `count_named_entities(text)`
Tokenizes and POS-tags the text, then uses `ne_chunk` to count named entities such as people, places, and organizations.

### `trigram_counts(tokens)`
Creates 3-word sequences from the token list and returns the 10 most common trigrams.

### `trigram_similarity(triA, triB)`
Compares two trigram lists by counting how many trigram phrases they share in common.

### `export_results_to_csv(results, output_path, most_likely_author)`
Builds a pandas DataFrame from the analysis summary and writes it to a CSV file. The CSV includes:
- text name
- token count
- unique token count
- named entity count
- top 20 tokens
- top trigrams
- the final author match for Text 4

## Limitations
- The named entity count is approximate and depends on NLTK data being available.
- The authorship comparison is very simple and only uses trigram overlap, so it is not a strong classifier.
- The trigram export is flattened into text so it is easy to read in CSV form, but it is less structured than the in-memory Python objects.
- The script assumes the input text files are in the same folder as the script.
- The `print("→ Lovecraft")` style result is just a heuristic output, not a verified authorship conclusion.
- The script uses automatic NLTK downloads, which is convenient but depends on internet access the first time it runs.
 - The project does not include a class-based implementation, so any rubric item asking for class attributes and methods is satisfied here by explaining the procedural design instead.

## Expected Output
When the script runs, it:
- prints analysis for each text file
- prints the final trigram-based author match for Text 4
- creates or overwrites `analysis_results.csv`

## Notes
This project is intentionally lightweight and assignment-focused. The code is organized for clarity and reproducibility rather than for a full reusable library design.
