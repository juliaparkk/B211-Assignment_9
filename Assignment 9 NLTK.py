import nltk
import collections
from pathlib import Path
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk, ngrams
from nltk.probability import FreqDist

# ---------------------------------------------------------
# Load Texts
# ---------------------------------------------------------

# Download the NLTK resources the script depends on
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("words", quiet=True)

def load_text(path):
    # Read each file as UTF-8 so punctuation and special characters are preserved.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Store all four texts in memory so the same processing pipeline can run on each one.
text1 = load_text("RJ_Lovecraft.txt")
text2 = load_text("RJ_Tolkein.txt")
text3 = load_text("RJ_Martin.txt")
text4 = load_text("Martin.txt")

# A dictionary keeps the text label paired with the content, which makes the loop below simpler.
texts = {
    "Text_1_Lovecraft": text1,
    "Text_2_Tolkein": text2,
    "Text_3_Martin": text3,
    "Text_4_Martin": text4
}

# ---------------------------------------------------------
# Preprocessing Functions
# ---------------------------------------------------------

stop_words = stopwords.words("english")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Tokenize, keep only alphabetic words, and remove common stop words.
    # This reduces noise so frequency counts focus on meaningful content words.
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def stem_tokens(tokens):
    # Reduce words to stems so related forms count as the same base term.
    # For example, "running" and "runs" both become easier to compare.
    return [stemmer.stem(t) for t in tokens]

def lemma_tokens(tokens):
    # Convert words to dictionary forms for a cleaner normalized view.
    # Lemmatization is more readable than stemming because it keeps real words.
    return [lemmatizer.lemmatize(t) for t in tokens]

# ---------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------

def count_named_entities(text):
    # Run POS tagging first because the chunker needs tagged tokens.
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    # `ne_chunk` groups names like people, places, and organizations.
    # Counting labeled chunks gives a rough measure of how many proper entities appear.
    chunked = ne_chunk(tagged, binary=False)

    count = 0
    for subtree in chunked:
        if hasattr(subtree, "label"):
            count += 1
    return count

# ---------------------------------------------------------
# N-gram Analysis (Trigrams)
# ---------------------------------------------------------

def trigram_counts(tokens):
    # Build 3-word sequences and return the most frequent ones.
    # Trigrams help capture repeated local phrasing instead of single-word frequency.
    tri = ngrams(tokens, 3)
    counter = collections.Counter(tri)
    return counter.most_common(10)

# ---------------------------------------------------------
# Run Analysis
# ---------------------------------------------------------

results = {}

for name, text in texts.items():
    # Process each text through the exact same pipeline so the outputs stay comparable.
    print("\n========================================")
    print(f"Processing {name}")
    print("========================================")

    tokens = preprocess(text)
    stems = stem_tokens(tokens)
    lemmas = lemma_tokens(tokens)

    # Frequency distribution shows which content words dominate each text.
    freq = FreqDist(tokens)
    top20 = freq.most_common(20)

    # Count named entities as a simple signal of how much proper-noun style content appears.
    ne_count = count_named_entities(text)

    # Trigrams capture repeated local word patterns in each text.
    tri = trigram_counts(tokens)

    # Save the results so the final comparison can reuse them without recalculating.
    results[name] = {
        "tokens": tokens,
        "top20": top20,
        "named_entities": ne_count,
        "trigrams": tri
    }

    print("\nTop 20 Tokens:")
    for w, c in top20:
        print(f"{w}: {c}")

    print(f"\nNamed Entity Count: {ne_count}")

    print("\nTop 10 Trigrams:")
    for gram, c in tri:
        print(gram, ":", c)

# ---------------------------------------------------------
# Authorship Comparison for Text_4
# ---------------------------------------------------------

def trigram_similarity(triA, triB):
    # Compare only the trigram phrases themselves, not their counts.
    # This turns each text into a set of repeated phrase patterns.
    setA = set([t[0] for t in triA])
    setB = set([t[0] for t in triB])
    return len(setA.intersection(setB))

def export_results_to_csv(results, output_path, most_likely_author):
    # Build a table of summary rows, then export it with pandas.
    rows = []
    for name, data in results.items():
        top20_tokens = "; ".join([f"{word}:{count}" for word, count in data["top20"]])
        top_trigrams = "; ".join([f"{' '.join(gram)}:{count}" for gram, count in data["trigrams"]])

        rows.append({
            "text_name": name,
            "token_count": len(data["tokens"]),
            "unique_tokens": len(set(data["tokens"])),
            "named_entities": data["named_entities"],
            "top20_tokens": top20_tokens,
            "top_trigrams": top_trigrams,
            "most_likely_author_match": most_likely_author if name == "Text_4_Martin" else ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

tri4 = results["Text_4_Martin"]["trigrams"]

# Compare Text 4 against each candidate text using shared trigram overlap.
sim1 = trigram_similarity(tri4, results["Text_1_Lovecraft"]["trigrams"])
sim2 = trigram_similarity(tri4, results["Text_2_Tolkein"]["trigrams"])
sim3 = trigram_similarity(tri4, results["Text_3_Martin"]["trigrams"])

print("\n========================================")
print("AUTHORSHIP COMPARISON (Trigram Overlap)")
print("========================================")
print(f"Similarity with Lovecraft: {sim1}")
print(f"Similarity with Tolkein:   {sim2}")
print(f"Similarity with RJ Martin: {sim3}")

print("\nMost likely author match for Text_4:")
# Pick the author with the highest shared trigram overlap.
if max(sim1, sim2, sim3) == sim1:
    most_likely_author = "Lovecraft"
elif max(sim1, sim2, sim3) == sim2:
    most_likely_author = "Tolkein"
else:
    most_likely_author = "Martin"

print(f"→ {most_likely_author}")

# Save the summary table next to the script so it can be opened outside Python.
csv_path = Path(__file__).with_name("analysis_results.csv")
export_results_to_csv(results, csv_path, most_likely_author)
print(f"\nCSV export written to: {csv_path}")
