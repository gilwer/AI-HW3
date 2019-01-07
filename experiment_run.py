import csv
import pickle
import random
from functools import reduce

from classifier import knn_factory, decision_factory, perceptron_factory
from hw3_utils import load_data


def split_crosscheck_groups(dataset, num_folds):
    ls = list(map(lambda x: ([(x[j])[0] for j in range(len(x))], [(x[k])[1] for k in range(len(x))]),
                  random_partition(list(map(lambda x, y: (x, y), dataset[0], dataset[1])), num_folds)))
    for i in range(num_folds):
        path = "ecg_fold_" + str(i+1) + ".data"
        pickle.dump(ls[i], open(path, "wb"))


def random_partition(lst, n):
    random.shuffle(lst)
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


def load_k_fold_data(i):
    path = "ecg_fold_" + str(i) + ".data"
    return pickle.load(open(path, "rb"))


def run_init():
    patiants,labels,test = load_data("data\Data.pickle")
    split_crosscheck_groups((patiants, labels), 2)

def evaluate(classifier_factory, k):
    a = [load_k_fold_data(i+1) for i in range(k)]
    num_correct = 0
    num_tests = 0
    for j in range(k):
        b = [a[i] for i in list(filter(lambda x: x !=j, range(k)))]
        data = reduce((lambda x, y: (x[0]+y[0], x[1] + y[1])), b)
        trainer = classifier_factory.train(data[0], data[1])
        result = list(map(lambda x: trainer.classify(x), a[j][0]))
        num_correct += sum(list(map(lambda x, y: x == y, result, a[j][1])))
        num_tests += len(result)
    return num_correct/num_tests, 1 - num_correct/num_tests

def run_additional_tests():
    accuracy_tree, errors_tree = evaluate(decision_factory(), 2)
    accuracy_percept, errors_percept = evaluate(perceptron_factory(), 2)
    with open('experiment12.csv', mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([1, accuracy_tree, errors_tree])
        writer.writerow([2, accuracy_percept, errors_percept])


def test_k():
    k = [1, 3, 5, 7, 13]
    for i in range(len(k)):
        accuracy, errors = evaluate(knn_factory(k[i]), 2)
        with open('knn_exp.csv', mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([k[i], accuracy, errors])



if __name__ == '__main__':
  #  test_k()
    run_additional_tests()