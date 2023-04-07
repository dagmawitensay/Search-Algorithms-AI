from collections import deque
import heapq
import math

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

if __name__=="__main__":
    print()
