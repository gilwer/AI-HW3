
from hw3_utils import abstract_classifier, abstract_classifier_factory
from submission_utils import euclidean_distance


class knn_classifier(abstract_classifier):

    def __init__(self, k, data, tags):
        self.k = k
        self.tags = tags
        self.data = data

    def classify(self, features):
        ls = sorted(range(len(self.tags)),
               key = lambda i: (list(map(lambda x: euclidean_distance(x, features), self.data)))[i])
        return sum([self.tags[ls[i]] for i in range(self.k)]) > (self.k/2)

class knn_factory(abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(self.k, data, labels)
