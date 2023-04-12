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
    
    def insertEdge(self, node_A, node_B, cost):
        """
        Insert Edge between node_A and node_B
        Args:
            node_A: first node
            node_B: second node
            cost: cost between node_A and node_B
        """

        if not node_A in self.graph:
            self.createNode(node_A)
        if not node_B in self.graph:
            self.createNode(node_B)
        
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