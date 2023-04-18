from Graph import Graph
import numpy as np
import networkx as nx


def adj_list_to_matrix(adj_list):
    nodes = list(adj_list.keys())
    n = len(nodes)
    adj_matrix = [[0 for i in range(n)] for j in range(n)]
    
    for node in adj_list:
        row_index = nodes.index(node)
        for neighbor in adj_list[node]:
            col_index = nodes.index(neighbor[0])
            adj_matrix[row_index][col_index] = 1
            
    return adj_matrix, nodes


def eigenvector_centrality(adj_matrix, tol=1e-6):
    n = len(adj_matrix)
    x = np.ones(n)
    x /= np.linalg.norm(x)

    while True:
        x_new = np.dot(adj_matrix, x)
        x_new /= np.linalg.norm(x_new)
        if np.abs(np.dot(x_new, x) - 1) < tol:
            break
        x = x_new

    return x_new


def katz_centrality(adj_matrix, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-6):
    n = len(adj_matrix)
    x = np.zeros(n)
    b = np.full(n, beta)
    I = np.identity(n)

    for _ in range(max_iter):
        x_new = alpha * np.dot(adj_matrix, x) + b
        err = np.linalg.norm(x_new - x)
        if err < tol:
            break

        x = x_new

    x /= np.linalg.norm(x)
    centrality = {i: x[i] for i in range(n)}

    return centrality


def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    M = np.array(M)
    N = M.shape[1]
    v = np.ones(N) / N
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v

def floyd_warshall(adj_matrix):
    n = len(adj_matrix)
    dist = [[float('inf') if i != j and adj_matrix[i][j] == 0 else adj_matrix[i][j] for j in range(n)] for i in range(n)]
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

def closeness_centrality(adj_matrix):
    dist_matrix = floyd_warshall(adj_matrix)
    n = len(adj_matrix)
    centralities = []
    
    for i in range(n):
        reachable_nodes = sum(1 for d in dist_matrix[i] if d != float('inf') and d != 0)
        total_distance = sum(d for d in dist_matrix[i] if d != float('inf') and d != 0)
        
        if total_distance == 0:
            centralities.append(0.0)
        else:
            centralities.append(reachable_nodes / total_distance)
    
    return centralities



graph = nx.Graph()
a = Graph()
a.insertEdge('Oradea','Sibiu', 151 )
a.insertEdge('Oradea','Zerind', 71)
a.insertEdge('Zerind','Arad', 75)
a.insertEdge('Arad','Sibiu', 140)
a.insertEdge('Arad','Timisoara', 118)
a.insertEdge('Sibiu','Fagaras', 99)
a.insertEdge('Sibiu','Rimnicu Vilcea', 80)
a.insertEdge('Timisoara','Lugoj', 111)
a.insertEdge('Lugoj','Mehadia', 70)
a.insertEdge('Mehadia','Drobeta', 75)
a.insertEdge('Drobeta','Craiova', 120)
a.insertEdge('Craiova','Pitesti', 138)
a.insertEdge('Rimnicu Vilcea','Pitesti', 97)
a.insertEdge('Rimnicu Vilcea', 'Craiova', 146)
a.insertEdge('Fagaras', 'Bucharest', 211)
a.insertEdge('Pitesti', 'Bucharest', 101)
a.insertEdge('Bucharest', 'Urziceni', 85)
a.insertEdge('Bucharest', 'Giurgiu', 90)
a.insertEdge('Urziceni', 'Hirsova', 98)
a.insertEdge('Urziceni', 'Vaslui', 142)
a.insertEdge('Hirsova', 'Eforie', 86)
a.insertEdge('Vaslui', 'Iasi', 92)
a.insertEdge('Iasi', 'Neamt', 87)
