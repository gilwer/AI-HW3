import pickle
import random


def split_crosscheck_groups(dataset, num_folds):
    ls = list(map(lambda x: ([(x[j])[0] for j in range(len(x))],[(x[k])[1] for k in range(len(x))]) ,random_partition(list(map(lambda x, y: (x, y), dataset[0],dataset[1])),num_folds)))
    for i in range(num_folds):
        path = "ecg_fold_" + str(i+1) + ".data"
        pickle.dump(ls[i], open(path, "wb"))


def random_partition(lst, n):
    random.shuffle(lst)
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


def load_k_fold_data(i):
    path = "ecg_fold_" + str(i+1) + ".data"
    return pickle.load(open(path, "wb"))
