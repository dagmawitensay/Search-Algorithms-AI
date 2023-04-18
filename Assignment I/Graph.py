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


def a_star_search(graph, start, goal):
    """
    Performs A* search algorithm to find the shortest path from start to goal using the given heuristic function h.
    """
    # Create an +
    96
    frontier = [(0, start)]
    # Create an empty set to store the visited nodes
    visited = set()
    # Create a dictionary to store the parent nodes
    parents = {}
    # Create a dictionary to store the cost from start to each node
    costs = {}
    # Set the cost of the start node to 0
    costs[start] = 0

    while frontier:
        # Get the node with the lowest priority
        current_cost, current_node = heapq.heappop(frontier)

        # Check if the current node is the goal node
        if current_node == goal:
            # If so, reconstruct the path from start to goal and return it
            path = [current_node]
            while current_node != start:
                current_node = parents[current_node]
                path.append(current_node)
            return path[::-1]

        # Add the current node to the visited set
        visited.add(current_node)

        # Expand the current node and add its neighbors to the frontier
        for neighbor in graph[current_node]:
            # Calculate the cost from start to the neighbor node
            tentative_cost = costs[current_node] + neighbor[1]

            # If the neighbor node has not been visited or the new cost is lower than the old cost
            if neighbor[0] not in visited or tentative_cost < costs.get(neighbor[0], float('inf')):
                # Update the cost and parent of the neighbor node
                costs[neighbor[0]] = tentative_cost
                parents[neighbor[0]] = current_node
                # Calculate the priority of the neighbor node using the heuristic function
                priority = tentative_cost + Heuristic(neighbor[0], goal)
                # Add the neighbor node with its priority to the frontier
                heapq.heappush(frontier, (priority, neighbor[0]))

    # If the goal node cannot be reached, return None
    return None

def aStarSearch(graph, start, goal):
    fringe = [(start, 0)]
    visited = set()
    costs = {}
    parents = {}
    costs[start] = 0

    while fringe:

        current_node, current_cost = heapq.heappop(fringe)
        if current_node == goal:
            path = [current_node]
            while current_node != start:
                current_node = parents[current_node]
                path.append(current_node)
            return path[::-1]

        for neighbor in graph[current_node]:
            tentetive_cost = costs[current_node] + neighbor[1]
            if neighbor[0] not in visited or tentetive_cost < costs.get(neighbor[0], float('inf')):
                visited.add(neighbor[0])
                costs[neighbor[0]] = tentetive_cost
                parents[neighbor[0]] = current_node

                priority = tentetive_cost + Heuristic(neighbor[0], goal)
                heapq.heappush(fringe, (neighbor[0], priority))
        
        return



if __name__=="__main__":       
    # graph = Graph()
    # graph.createNode(1)
    # graph.createNode(2)
    # graph.createNode(3)
    # graph.createNode(4)
    # graph.createNode(4)
    # graph.createNode(5)
    # graph.insertEdge(1,2)
    # graph.insertEdge(1, 3)
    # graph.insertEdge(2, 4)
    # graph.insertEdge(3, 4)
    # graph.insertEdge(4, 5)

    # print(graph.graph)
    # graph.deleteNode(1)
    # print(graph.graph)
    # graph.breadthFirstSearch()
    # graph.depthFirstSearch()
    # graph1 = {
    #     1: [2, 3],
    #     2: [1, 4],
    #     3: [1, 4],
    #     4: [2, 3, 5],
    #     5: [4]
    # }
    # aStarSearch(cityData, 4)
    graph = Graph()
    graph.createNode('Arad')
    graph.createNode('Oradea')
    graph.createNode('Zerind')
    graph.createNode('Timisoara')
    graph.createNode('Lujog')
    graph.createNode('Mehadia')
    graph.createNode('Drobeta')
    graph.createNode('Craiova')
    graph.createNode('Pitesti')
    graph.createNode('Rimincu Vielca')
    graph.createNode('Sibiu')
    graph.createNode('BFagaras')
    graph.createNode('Buckherest')
    graph.createNode('Giurgiu')
    graph.createNode('Urziceni')
    graph.createNode('Efroie')
    graph.createNode('Hirsova')
    graph.createNode('Vaslui')
    graph.createNode('Iasi')
    graph.createNode('Nemat')
    graph.insertEdge('Arad','Zerind', 75)
    graph.insertEdge('Arad', 'Timisoara', 118)
    graph.insertEdge('Arad', 'Sibiu', 140)
