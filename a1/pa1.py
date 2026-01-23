import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

stemmer = PorterStemmer()


def stemmed_tokenizer(text: str) -> list[str]:
    return [stemmer.stem(w) for w in wordpunct_tokenize(text.lower())]


PREPROCESSING_CONFIGS = [
    {
        "name": "count, no stopwords, unigrams",
        "vectorizer_class": CountVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": None,
            "ngram_range": (1, 1),
        },
    },
    {
        "name": "count, stopwords removed, unigrams",
        "vectorizer_class": CountVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 1),
        },
    },
    {
        "name": "count, no stopwords, uni+bigrams",
        "vectorizer_class": CountVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": None,
            "ngram_range": (1, 2),
        },
    },
    {
        "name": "count, stemmed, unigrams",
        "vectorizer_class": CountVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": None,
            "tokenizer": stemmed_tokenizer,
            "token_pattern": None,
            "ngram_range": (1, 1),
        },
    },
    {
        "name": "TF-IDF, no stopwords, unigrams",
        "vectorizer_class": TfidfVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": None,
            "ngram_range": (1, 1),
        },
    },
    {
        "name": "TF-IDF, stopwords removed, unigrams",
        "vectorizer_class": TfidfVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": (1, 1),
        },
    },
    {
        "name": "TF-IDF, no stopwords, uni+bigrams",
        "vectorizer_class": TfidfVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": None,
            "ngram_range": (1, 2),
        },
    },
    {
        "name": "tfidf, stemmed, unigrams",
        "vectorizer_class": TfidfVectorizer,
        "params": {
            "lowercase": True,
            "stop_words": None,
            "tokenizer": stemmed_tokenizer,
            "token_pattern": None,
            "ngram_range": (1, 1),
        },
    },
]


CLASSIFIERS = [
    ("naive bayes", MultinomialNB()),
    ("logistic regression", LogisticRegression(max_iter=1000)),
]


def load_dataset(file_class0: str, file_class1: str) -> tuple[list[str], np.ndarray]:
    sentences = []
    labels = []

    with open(file_class0, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                labels.append(0)

    with open(file_class1, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                labels.append(1)

    return sentences, np.array(labels)


def run_experiments(
    sentences: list[str], labels: np.ndarray, dataset_name: str
) -> list[dict]:
    results = []

    for cfg in PREPROCESSING_CONFIGS:
        vectorizer = cfg["vectorizer_class"](**cfg["params"])
        X = vectorizer.fit_transform(sentences)

        for classifier_name, classifier in CLASSIFIERS:
            scores = cross_val_score(classifier, X, labels, cv=5, scoring="accuracy")

            results.append(
                {
                    "dataset": dataset_name,
                    "preprocessing": cfg["name"],
                    "classifier": classifier_name,
                    "accuracy_mean": scores.mean(),
                    "accuracy_std": scores.std(),
                }
            )

    return results


def print_results(results: list[dict]):
    print(f"{'preprocessing':<40} {'classifier':<20} {'accuracy':<15}")
    print("-" * 75)

    for r in results:
        acc_str = f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}"
        print(f"{r['preprocessing']:<40} {r['classifier']:<20} {acc_str:<15}")


def main():
    print("loading dataset 1: sentiment classification (semantics)")

    sentences_synsem, labels_synsem = load_dataset("synsem0.txt", "synsem1.txt")
    print("loaded\n")

    results_synsem = run_experiments(sentences_synsem, labels_synsem, "sentiment")
    print_results(results_synsem)
    print("done\n")

    print("loading dataset 2: double letters classification (spelling)")
    sentences_morph, labels_morph = load_dataset("morphphon0.txt", "morphphon1.txt")
    print("loaded\n")

    results_morph = run_experiments(sentences_morph, labels_morph, "double letters")
    print_results(results_morph)
    print("done\n")


if __name__ == "__main__":
    main()
