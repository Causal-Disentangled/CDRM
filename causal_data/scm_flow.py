import pandas as pd
import os.path as osp
import inspect

import itertools
from itertools import combinations, chain
from scipy.stats import norm, pearsonr
import pandas as pd
import numpy as np
import math
import os
import argparse

from utils.scm_utils import plot


def skeleton(suffStat, indepTest, alpha, labels):
    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]

    G = []
    for i in range(len(labels)):
        G.append([])
        for l in range(len(labels)):
            G[i].append(True)
    for i in range(len(labels)):
        G[i][i] = False

    done = False
    ord = 0
    warning_point = []
    while done == False:
        done = True
        node_index = []


        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == True:
                    node_index.append((i, j))

        for x, y in node_index:
            if G[x][y] == True and ord == 0:
                done = False
                pval = indepTest(suffStat, x, y, [])
                if pval >= alpha:
                    G[x][y] = G[y][x] = False
            if G[x][y] == True and ord != 0:
                neighborsBool = G[x][:]
                neighborsBool[y] = False
                neighbor = []
                for i in range(len(neighborsBool)):
                    if neighborsBool[i] == True:
                        neighbor.append(i)
                for neighbors in set(itertools.combinations(neighbor, ord)):
                    pval = indepTest(suffStat, x, y, list(neighbors))
                    if pval != 0 and pval < alpha:
                        point = (x, y)
                        warning_point.append(point)
                        break
                    if pval >= alpha:
                        G[x][y] = G[y][x] = False
                        sepset[x][y] = list(neighbor)

        ord += 1
    return {'sk': np.array(G), 'sepset': sepset, 'point': warning_point}

def extend_cpdag(graph):
    def rule1(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[0], x[1])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)

            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and ((a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):

                        continue
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2

        return pdag

    def rule2(pdag, solve_conf=False):
        search_pdag = pdag.copy()
        ind = []

        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))


        for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2

        return pdag

    def rule3(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))


        for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)

            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:

                        if 'unfTriples' in graph.keys() and ((c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                            continue
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break

        return pdag

    def rule4(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[0], x[1])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 0) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)

            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and ((a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):

                        continue
                    if pdag[a][b] == 1 and pdag[b][a] == 1:
                        pdag[a][b] = 1
                        pdag[b][a] = 0


        return pdag

    i, j = graph['sk'].shape
    pdag = [[1 if graph['sk'][i][j] == True else 0 for i in range(i)] for j in range(j)]

    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j] == 1:
                ind.append((i, j))

    for i in range(len(ind)):
        x, y = ind[i][0], ind[i][1]
        for z in range(len(pdag)):
            if graph['sk'][y][z] == True and graph['sk'][x][z] == False and z != x and \
                    not (y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0

    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)
    pdag = rule4(pdag)

    return np.array(pdag)

def extend_dag(cpdag):
    def rule1(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[0], x[1])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)

            if len(isC) > 0:
                for c in isC:
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2

        return pdag

    def rule2(pdag, solve_conf=False):
        search_pdag = pdag.copy()
        ind = []

        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))


        for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2

        return pdag

    def rule3(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)

            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break

        return pdag

    def rule4(pdag, solve_conf=False, unfVect=None):
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        for a, b in sorted(ind, key=lambda x:(x[0], x[1])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 0) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)

            if len(isC) > 0:
                for c in isC:
                    if pdag[a][b] == 1 and pdag[b][a] == 1:
                        pdag[a][b] = 1
                        pdag[b][a] = 0


        return pdag


    pdag = cpdag

    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)
    pdag = rule4(pdag)

    return np.array(pdag)

def pc(suffStat, alpha, labels, indepTest):
    graphDict = skeleton(suffStat, indepTest, alpha, labels)
    cpdag = extend_cpdag(graphDict)
    ind = []
    war= []
    for i in range(len(cpdag)):
        for j in range(len(cpdag[i])):
            if cpdag[i][j] == True:
                ind.append((i, j))
    for i in range(len(ind)):
        for l in range(len(graphDict['point'])):
            if ind[i] == graphDict['point'][l]:
                war.append(ind[i])
    for i in range(len(war)):
        x, y = war[i]
        cpdag[x][y] = 0
        cpdag[y][x] = 0
    dag = extend_dag(cpdag)
    return dag

def gauss_ci_test(suffstat, x, y, S):
    C = suffstat["C"]
    n = suffstat["n"]
    cut_at = 0.9999999
    if len(S) == 0:
        r = C[x, y]
    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))
    else:
        m = C[np.ix_([x]+[y] + S, [x] + [y] + S)]
        PM = np.linalg.pinv(m)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    r = min(cut_at, max(-1 * cut_at, r))
    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    return 2 * (1 - norm.cdf(abs(res)))

def write_dag(scm, args):
    data_dir = './causal_data/flow/{}_{}_{}/'.format(args.log_dir, args.ratio, args.epochs)
    #data_dir = './causal_data/flow/1/' #test
    fname = 'dag'
    data_path = os.path.join(data_dir, fname + '.txt')
    row, col = scm.shape
    for i in range(row):
        a = float(scm[i, 0])
        b = float(scm[i, 1])
        c = float(scm[i, 2])
        d = float(scm[i, 3])
        with open(data_path, 'a') as f:
            f.write(str(int(a))+" "+str(int(b))+" "+str(int(c))+" "+str(int(d))+'\n')

def dag(args):
#if __name__ == '__main__':#test
    image_path = './causal_data/flow/{}_{}_{}/scm.png'.format(args.log_dir, args.ratio, args.epochs)
    #image_path = './causal_data/flow/1/scm.png'#test
    flow_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    #df_np = np.loadtxt(flow_path + '\\causal_data\\flow\\1\\restore_flow_2000.txt')#test
    df_np = np.loadtxt(flow_path + '/causal_data/flow/{}_{}_{}/restore_flow_{}.txt'.format(args.log_dir, args.ratio, args.epochs,args.epochs))
    #df_np = np.loadtxt(flow_path + '\\causal_data\\flow_noise\\flow.txt')#ori
    columns = ['X', 'Y', 'Z', 'W']
    df = pd.DataFrame(df_np, columns=columns)
    #df_X = pd.DataFrame(df_np[:, :-2])
    row, col = df.shape
    scm = pc(
        suffStat={"C": df.corr().values, "n": df.values.shape[0]},
        alpha=0.01,
        labels=[str(i) for i in range(col)],
        indepTest=gauss_ci_test,
    )
    plot(scm, columns, image_path)
    write_dag(scm, args)
    #print(scm)
    return scm