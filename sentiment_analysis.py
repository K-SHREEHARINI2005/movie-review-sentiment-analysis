import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Download necessary datasets (only first time)
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')

# Load documents (words + category)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Feature extractor
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in words)
    return features

# Create feature sets
featuresets = [(document_features(d), c) for (d,c) in documents]

# Train-test split
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate
print("Accuracy:", accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
