from collections import deque
import heapq
import math
import random
import numpy as np

class Node:
    """
    Node to represent city in hand
    """

class Graph:
    """
    Graph class that defines a graph data structure
    """
    def __init__(self):
        self.graph = {}
    
    def createNode(self, node, cost=1):
        """
        Create a node in the graph
        """
        self.graph[node] = []
    
    def insertEdge(self, node_A, node_B, cost):
        """
        Insert Edge between node_A and node_B
        """
        if node_A not in self.graph:
            self.createNode(node_A)
        if node_B not in self.graph:
            self.createNode(node_B)
        self.graph[node_A].append((node_B, cost))
        self.graph[node_B].append((node_A, cost))
    
    def deleteEdge(self, node_A, node_B, cost):
        """
        delete edge between node node_A and node_B
        """
        self.graph[node_A].remove((node_B, cost))
        self.graph[node_B].remove((node_A, cost))

    def deleteNode(self, node_A):
        """
        delete node from graph
        """
        for node in self.graph:
            for neighbor in self.graph[node]:
                if neighbor[0] == node_A:
                    self.graph[node].remove(neighbor)
        
        del self.graph[node_A]

    def depthFirstSearch(self):
        visited = set()
        self.depthFirstHelper(list(self.graph.keys())[0], visited)

    def depthFirstHelper(self, start, visited):
        """
        A function that traverses the graph in depth first manner
        """
        visited.add(start)
        for neigbor in self.graph[start]:
            if neigbor[0] not in visited:
                visited.add(neigbor[0])
                self.depthFirstHelper(neigbor[0], visited)
        
        return 
    
    def breadthFirstSearch(self):
        """
        A function that traverses the graph in breadth first manner
        """
        start = list(self.graph.keys())[0]
        visited = {start,}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neigbor in self.graph[current]:
                if neigbor[0] not in visited:
                    visited.add(neigbor[0])
                    queue.append(neigbor[0])
        return
    
    def uniformCostSearch(self):
        pass


cityData = {}
with open("./citys.txt","r") as file:
    for line in file:
        line = line.split()
        cityData[line[0]] = (line[1],line[2])
del cityData['City']

def Heuristic(node_A, node_B):
    """"
    calculates the distance between the two cities in km using Haversine formula
    """
    long1, lat1, long2, lat2 = map(math.radians, [float(cityData[node_A][1]), float(cityData[node_A][0]),float(cityData[node_B][0]) * (math.pi / 180), float(cityData[node_B][0])])

    # Haversine formula
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371      # Radius of earth in kilometers. Use 3956 for miles
    # calculate the result
    distance = c * r

    print(distance)


def aStarSearch(graph, goal, h):
    start = list(graph.graph.keys())[0]
    fringe = [(start, 0)]
    costs = {}
    parent = {}

    while fringe:
        current_node = heapq.heapop(fringe)
        
        if parent == goal:
            path = []
            while current_node in path:
                current
def a_star_search(graph, start, end):
    def reconstruct_path(came_from, current_node):
        path = [current_node]
        while current_node in came_from:
            current_node = came_from[current_node]
            path.append(current_node)
        return path[::-1]  # Reverse the path

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = haversine(start[0], start[1], end[0], end[1])

    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == end:
            return reconstruct_path(came_from, current)

        for neighbor, cost in graph[current].items():
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + haversine(neighbor[0], neighbor[1], end[0], end[1])
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


def katz_centrality(adj_list, alpha=0.1, beta=1, max_iter=1000, tol=1e-6):
    """
    Compute Katz centrality for a graph represented as an adjacency list.

    Parameters:
    adj_list (dict): The adjacency list of the graph.
    alpha (float): The damping factor for indirect connections.
    beta (float): The scaling factor for the centrality scores.
    max_iter (int): The maximum number of iterations for the power method.
    tol (float): The tolerance for convergence.

    Returns:
    A dictionary of centrality scores, where the keys are node IDs and the values are the centrality scores.
    """
    n = len(adj_list)
    A = np.zeros((n, n))
    for i, neighbors in adj_list.items():
        for j in neighbors[0]:
            A[i-1][j-1] = 1
    print(A)
    # calculate the centrality scores using the power method
    x = np.ones(n)
    centrality = np.zeros(n)
    for i in range(max_iter):
        centrality = beta * np.matmul(A, x) + alpha * x
        if np.allclose(x, centrality, atol=tol):
            break
        x = centrality

    # normalize the centrality scores
    centrality /= np.max(centrality)

    # return the centrality scores as a dictionary
    centrality_dict = {}
    for i, score in enumerate(centrality):
        centrality_dict[i+1] = score
    return centrality_dict


if __name__=="__main__":
    print(cityData)


def generate_random_graph(n, p):
        graph = Graph()
        nodes = list(range(n))
        for i in range(n):
            for j in range(n):
                if random.random() < p:
                    graph.insertEdge(i, j, 1)
        return graph
graph1 = generate_random_graph(10, 0.2)
print(graph1.graph)

