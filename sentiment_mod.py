import nltk
import random
from nltk import word_tokenize
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


short_pos = open("Dataset/short_reviews/positive.txt","r",encoding='Windows-1252').read()
short_neg = open("Dataset/short_reviews/negative.txt","r",encoding='Windows-1252').read()

all_words = []
documents = []

# j is adject, r is adverb, and v is verb
# allowed_word_types = ["J", "R", "V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# save_documents = open("pickled_algos/documents.pickle","wb")
# pickle.dump(documents, save_documents)
# save_documents.close()

p_f = open("pickled_algos/documents.pickle","rb")
documents = pickle.load(p_f)
p_f.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

# save_word_features = open("pickled_algos/word_features5k.pickle","wb")
# pickle.dump(word_features, save_word_features)
# save_word_features.close()

p_f = open("pickled_algos/word_features5k.pickle","rb")
word_features = pickle.load(p_f)
p_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
#print((find_features((movie_reviews.words('neg/cv000_29416.txt')))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]
print(len(featuresets)) #10644

random.shuffle(featuresets)

"""
Naive Bayes
"""

training_set = featuresets[:10000]
testing_set = featuresets[10000:]



# classifier = nltk.NaiveBayesClassifier.train(training_set)
#
# print("Naive Bayes Algo accuracy", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(15)
#
# """
# Pickle save classifier
# """
# save_classifier = open("pickled_algos/Twitter_naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

classifier_f = open("pickled_algos/Twitter_naivebayes.pickle","rb")
Naivebayes_classifier = pickle.load(classifier_f)
classifier_f.close()

# print("Naive Bayes Algo accuracy", (nltk.classify.accuracy(Naivebayes_classifier, testing_set))*100)
# classifier.show_most_informative_features(15)

"""
Scikit-Learn incorporation
"""
# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB_classifier accuracy", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
#
# save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

classifier_f = open("pickled_algos/MNB_classifier5k.pickle","rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()


# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# print("BernoulliNB_classifier accuracy", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
#
#
# save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()

classifier_f = open("pickled_algos/BernoulliNB_classifier5k.pickle","rb")
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()




# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier_classifier accuracy", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
#
# save_classifier = open("pickled_algos/SGDClassifier_classifier5k.pickle","wb")
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()

classifier_f = open("pickled_algos/SGDClassifier_classifier5k.pickle","rb")
SGDClassifier_classifier = pickle.load(classifier_f)
classifier_f.close()

# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print("LinearSVC_classifier accuracy", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
#
# save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()

classifier_f = open("pickled_algos/LinearSVC_classifier5k.pickle","rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

"""
Combining Algos with a Vote
"""
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.classifiers = classifiers
    def classify(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


voted_classifiers = VoteClassifier(Naivebayes_classifier,MNB_classifier,BernoulliNB_classifier,
                                   SGDClassifier_classifier,LinearSVC_classifier)
# print("voted_classifiers accuracy", (nltk.classify.accuracy(voted_classifiers, testing_set))*100)
# print("Classification:", voted_classifiers.classify(testing_set[0][0]), "Confidence {:f} %".format(voted_classifiers.confidence(testing_set[0][0])*100))

def sentiment(text):
    feats = find_features(text)
    return (voted_classifiers.classify(feats),voted_classifiers.confidence(feats))