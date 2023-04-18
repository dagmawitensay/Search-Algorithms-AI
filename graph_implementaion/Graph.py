import random
import math
class Graph:
    """
    Graph class that defines a graph data structure
    """
    def __init__(self):
        self.graph = {}
        self.locations = {}

    def createNode(self, node):
        """
        Create a node in the graph
        Args:
            node: value of node of creation
        """
        self.graph[node] = []
        self.locations[node] = (random.uniform(1.0, 25) , random.uniform(1.0, 25))
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371  # radius of Earth in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c
        return d

    def insertEdge(self, node_A, node_B, cost):
        """
        Insert Edge between node_A and node_B
        Args:
            node_A: first node
            node_B: second node
            cost: cost between node_A and node_B
        """

        if node_A not in self.graph:
            self.createNode(node_A)
        if node_B not in self.graph:
            self.createNode(node_B)
        
        if cost == 1:
            cost = int(self.haversine_distance(self.locations[node_A][0], self.locations[node_A][1], self.locations[node_B][0], self.locations[node_B][1]))
        
        self.graph[node_A].append((node_B, cost))
        self.graph[node_B].append((node_A, cost))
    
    def deleteEdge(self, node_A, node_B, cost):
        """
        delete edge between node node_A and node_B
        Args:
            node_A: first node
            node_B: second node
            cost: cost between node_A and node_B
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
    
    def neighbours(self,nod):
        if nod in self.graph and self.graph[nod] != None:
            return self.graph[nod]