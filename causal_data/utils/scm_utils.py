import networkx as nx
import matplotlib.pyplot as plt

def plot(graph, labels: list, path: str):
    """可视化学习出的贝叶斯网络"""
    G = nx.DiGraph()
    for i in range(len(graph)):
        G.add_node(labels[i])
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                G.add_edges_from([(labels[i], labels[j])])
    nx.draw(G, with_labels=True)
    plt.savefig(path)
    plt.show()
