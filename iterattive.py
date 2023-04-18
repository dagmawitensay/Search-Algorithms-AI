class Graph:
    """
    Graph class that defines a graph data structure
    """

    def __init__(self):
        self.graph = {}
        self.locations = {}

    def createNode(self, node, latitude, longitude):
        """
        Create a node in the graph
        Args:
            node: value of node of creation
        """
        if node in self.graph:
            raise ValueError("Node already exists in the graph.")
        self.graph[node] = []
        self.locations[node] = (latitude, longitude)

    def insertEdge(self, node_A, node_B, cost):
        """
        Insert Edge between node_A and node_B
        Args:
            node_A: first node
            node_B: second node
            cost: cost between node_A and node_B
        """

        if not node_A in self.graph:
            self.createNode(node_A, None, None)
        if not node_B in self.graph:
            self.createNode(node_B, None, None)

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

    def iterative_deepening_search(self, graph, start, goal, max_depth):
        """
        Iterative Deepening Search (IDS) algorithm for city search based on latitude and longitude coordinates.

        Args:
            graph (dict): Graph representation with city nodes as keys and latitude/longitude coordinates as values.
            start (str): Start city node.
            goal (str): Goal city node.
            max_depth (int): Maximum depth to explore in the search.

        Returns:
            list: List of city nodes representing the path from start to goal, or None if no path is found.
        """
        for depth in range(max_depth + 1):
            visited = set()
            path = self.dfs(graph, start, goal, depth, visited)
            if path is not None:
                return path
        return None

    def dfs(self, graph, node, goal, depth, visited):
        """
        Depth-First Search (DFS) function for city search based on latitude and longitude coordinates.

        Args:
            graph (dict): Graph representation with city nodes as keys and latitude/longitude coordinates as values.
            node (str): Current city node.
            goal (str): Goal city node.
            depth (int): Remaining depth to explore in the search.
            visited (set): Set of visited city nodes.

        Returns:
            list: List of city nodes representing the path from start to goal, or None if no path is found.
        """
        if node == goal:
            return [node]
        if depth == 0:
            return None

        visited.add(node)

        for neighbor, cost in self.graph[node]:
            if neighbor not in visited:
                path = self.dfs(graph, neighbor, goal, depth - 1, visited)
                if path is not None:
                    return [node] + path

        return None


if __name__ == "__main__":
    a = Graph()
    a.insertEdge('Oradea', 'Sibiu', 151)
    a.insertEdge('Oradea', 'Zerind', 71)
    a.insertEdge('Zerind', 'Arad', 75)
    a.insertEdge('Arad', 'Sibiu', 140)
    a.insertEdge('Arad', 'Timisoara', 118)
    a.insertEdge('Sibiu', 'Fagaras', 99)
    a.insertEdge('Sibiu', 'Rimnicu Vilcea', 80)
    a.insertEdge('Timisoara', 'Lugoj', 111)
    a.insertEdge('Lugoj', 'Mehadia', 70)
    a.insertEdge('Mehadia', 'Drobeta', 75)
    a.insertEdge('Drobeta', 'Craiova', 120)
    a.insertEdge('Craiova', 'Pitesti', 138)
    a.insertEdge('Rimnicu Vilcea', 'Pitesti', 97)
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
    start = 'Oradea'
    goal = 'Giurgiu'
    depth_limit = 4
    path = a.iterative_deepening_search(a.graph, start, goal, depth_limit)
    print(path)
    # if path is None:
    #     print("No valid path found within depth limit.")
    # else:
    #     print("Path from {} to {}: {}".format(start, goal, "->".join(path)))
