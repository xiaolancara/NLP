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

documents = []

for r in short_pos.split('\n'):
    documents.append((r, "pos"))
for r in short_neg.split('\n'):
    documents.append((r, "neg"))
all_words = []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)
for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# top 15 most common words
# print(all_words.most_common(15))
# print(all_words["stupid"])

word_features = list(all_words.keys())[:5000]
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
# save_classifier = open("Twitter_naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

classifier_f = open("pickled_algos/Twitter_naivebayes.pickle","rb")
Naivebayes_classifier = pickle.load(classifier_f)
classifier_f.close()

print("Naive Bayes Algo accuracy", (nltk.classify.accuracy(Naivebayes_classifier, testing_set))*100)
classifier.show_most_informative_features(15)

"""
Scikit-Learn incorporation
"""
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


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
print("voted_classifiers accuracy", (nltk.classify.accuracy(voted_classifiers, testing_set))*100)
print("Classification:", voted_classifiers.classify(testing_set[0][0]), "Confidence {:f} %".format(voted_classifiers.confidence(testing_set[0][0])*100))