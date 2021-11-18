import random
import time
import numpy as np
import os
from models.metric import *


def GetHyper(meta, hyp, txt_path='evolve.txt') -> float:
    if(os.path.isfile(txt_path)):
        x = np.loadtxt(txt_path, ndmin=2)  # 读取超参数组合集
        n = min(5, len(x))  # number of previous results to consider
        x = x[np.argsort(-fitness(x))][:n]  # top n mutations
        w = fitness(x) - fitness(x).min()  # weights

        parent = 'single'
        if parent == 'single' or len(x) == 1:
            x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
        elif parent == 'weighted':
            x = (x*w.reshape(n, 1)).sum(0)/w.sum(0)  # weighted combination

        # Mutate
        mp, s = 0.8, 0.2  # mutation probability, sigma
        npr = np.random
        npr.seed(int(time.time()))
        g = np.array([x[0] for x in meta.values()])  # gains 0-1
        ng = len(meta)
        v = np.ones(ng)
        while all(v == 1):  # mutate until a change occurs (prevent duplicates)
            v = (g * (npr.random(ng) < mp) * npr.randn(ng)
                 * npr.random() * s + 1).clip(0.3, 3.0)
        for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
            hyp[k] = float(x[i+4] * v[i])  # mutate

    # Constrain to limits
    for k, v in meta.items():
        hyp[k] = max(hyp[k], v[1])  # lower limit
        hyp[k] = min(hyp[k], v[2])  # upper limit
        hyp[k] = round(hyp[k], 5)  # significant digits

    return hyp['lr'], hyp['positive_weight']


def Update_gene(hyp, metric, txt_path='evolve.txt') -> None:
    # Print the metric
    a = '%20s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%20.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%20s' * len(metric) % tuple(metric.keys())  # metric keys
    d = '%20.3g' * len(metric) % tuple(metric.values())  # metric values
    print('\nHyperparameters:\n%s\n%s\nEvolved fitness:\n %s\n%s\n\n' % (a, b, c, d))

    # Save in txt_path
    with open(txt_path, 'a') as f:  # append result
        f.write(d + b + '\n')
    x = np.unique(np.loadtxt(txt_path, ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt(txt_path, x, '%10.3g')  # save sort by fitness


def fitness(x):
    # Model fitness as a weighted combination of metrics
    # weights for [acc, precision, recall, Fscore]
    w = [0, 0.15, 0.15, 0.7]
    return (x[:, :4] * w).sum(1)
