import numpy as np
import math


X = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9]
Y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0],dtype=np.float64)
N = len(X)
weight = [1/N for i in range(N)]

def setthreshold(X, Y, weight):
    prob_error = []
    for i in range(N):
        temp = []
        error = 0
        i += 0.5
        for j in X:
            if j <= i:
                temp.append(1.0)
            else:
                temp.append(-1.0)
        for j in range(N):
            if Y[j] != temp[j]:
                error += weight[j]
        prob_error.append(error)
    return min(prob_error), prob_error.index(min(prob_error))

def negative_setthreshold(X, Y, weight):
    prob_error = []
    for i in range(N):
        temp = []
        error = 0
        i += 0.5
        for j in X:
            if j <= i:
                temp.append(-1.0)
            else:
                temp.append(1.0)
        for j in range(N):
            if Y[j] != temp[j]:
                error += weight[j]
        prob_error.append(error)
    return min(prob_error), prob_error.index(min(prob_error))

def weakclassify(X, v):
    G = []
    for i in X:
        if i <= v:
            G.append(1.0)
        else:
            G.append(-1.0)
    return G

def neagtive_weakclassify(X, v):
    G = []
    for i in X:
        if i <= v:
            G.append(-1.0)
        else:
            G.append(1.0)
    return G

def set_alpham(prob_error):
    return 1/2 * math.log((1 - prob_error) / prob_error, math.e)

def set_Zm(weight, alpham, G, Y):
    Zm = 0.0
    for i in range(N):
        Zm += weight[i] * math.e**(-alpham * Y[i] * G[i])
    return Zm

def update_weight(Zm, weight, alpham, G, Y):
    for i in range(N):
        weight[i] = weight[i] / Zm * math.e**(-alpham * Y[i] * G[i])
    return weight

def AdaBoost(X, Y, weight):
    prob_error_1 = setthreshold(X, Y, weight)
    prob_error_2 = negative_setthreshold(X, Y, weight)
    if prob_error_1[0] < prob_error_2[0]:
        prob_error = prob_error_1[0]
        v = prob_error_1[1]
        G = weakclassify(X, v)
    else:
        prob_error = prob_error_2[0]
        v = prob_error_2[1]
        G = neagtive_weakclassify(X, v)
    k = 0

    alpham = set_alpham(prob_error)
    print('弱分类器%sG(x)'%alpham)
    weak = alpham * np.array(G)
    Gx = np.sign(weak)
    index = 1
    m = 1
    for i in range(2):
        Zm = set_Zm(weight, alpham, G, Y)
        weight = update_weight(Zm, weight, alpham, G, Y)
        prob_error_1 = setthreshold(X, Y, weight)
        prob_error_2 = negative_setthreshold(X, Y, weight)
        if prob_error_1[0] < prob_error_2[0]:
            prob_error = prob_error_1[0]
            v = prob_error_1[1]
            G = weakclassify(X, v)
        else:
            prob_error = prob_error_2[0]
            v = prob_error_2[1]
            G = neagtive_weakclassify(X, v)
        alpham = set_alpham(prob_error)
        print('弱分类器%sG(x)' % alpham)
        weak += alpham*np.array(G)
        m += 1
        Gx = np.sign(weak)




AdaBoost(X, Y, weight)



