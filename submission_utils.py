

def sqrt(x):
    return x**(1/2)


def euclidean_distance(row1, row2):
    return sqrt(sum(list(map(lambda x, y: (x-y)**2, row1, row2))))


