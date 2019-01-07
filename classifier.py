
from hw3_utils import abstract_classifier, abstract_classifier_factory
from submission_utils import euclidean_distance
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import tree

class knn_classifier(abstract_classifier):

    def __init__(self, k, data, tags):
        self.k = k
        self.tags = tags
        self.data = data

    def classify(self, features):
        key_val = (list(map(lambda x: euclidean_distance(x, features), self.data)))
        ls = sorted(range(len(self.tags)),
               key = lambda i: key_val[i])
        return sum([self.tags[ls[i]] for i in range(self.k)]) > (self.k/2)

class knn_factory(abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(self.k, data, labels)


class decision_factory(abstract_classifier_factory):

    def train(self, data, labels):
        return decision_tree_classifier(DecisionTreeClassifier().fit(data, labels))


class decision_tree_classifier(abstract_classifier):

    def __init__(self, tree_classifier):
       self.tree_classifier = tree_classifier

    def classify(self, features):
        return self.tree_classifier.predict(features)

class perceptron_classifier(abstract_classifier):

    def __init__(self, precption):
        self.precption = precption

    def classify(self, features):
        return self.precption.predict(features)


class perceptron_factory(abstract_classifier_factory):

    def train(self, data, labels):
        return perceptron_classifier(linear_model.Perceptron().fit(data, labels))
