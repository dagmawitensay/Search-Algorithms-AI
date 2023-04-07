from collections import deque
from Graph import Graph
import heapq
import math


def depthFirstSearch(graph):
    visited = set()
    depthFirstHelper(graph, list(graph.keys())[0], visited)

def depthFirstHelper(graph, start, visited):
    """
    A function that traverses the graph in depth first manner
    """
    visited.add(start)
    print(start)
    for neigbor in graph[start]:
        if neigbor[0] not in visited:
            visited.add(neigbor[0])
            depthFirstHelper(graph, neigbor[0], visited)
    
    return 

def breadthFirstSearch(graph):
    """
    A function that traverses the graph in breadth first manner
    """
    start = list(graph.keys())[0]
    visited = {start,}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        print(current)
        for neigbor in graph[current]:
            if neigbor[0] not in visited:
                visited.add(neigbor[0])
                queue.append(neigbor[0])
    return

def uniformCostSearch(self):
    pass

Data = {}
with open("./citys.txt","r") as file:
    for line in file:
        line = line.split()
        if len(line) > 3:
            Data[f"{line[0]} {line[1]}"] = (line[2], line[3])
        else:
            Data[line[0]] = (line[1],line[2])
del Data['City']

def Heuristic(node_A, node_B, cityData):
    """"
    calculates the distance between the two cities in km using Haversine formula
    """
    long1, lat1, long2, lat2 = map(math.radians, [float(cityData[node_A][1]), float(cityData[node_A][0]),float(cityData[node_B][0]) * (math.pi / 180), float(cityData[node_B][0])])

    # Haversine formula
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371      # Radius of earth in kilometers.
    distance = c * r

    return distance


def astar(start_node, end_node, graph, lat_long_dict):
    fringe = []
    visited = set()
    parent = {}
    actual_cost = {start_node: 0}
    total_estimated_cost = {start_node: Heuristic(start_node, end_node, lat_long_dict)}

    heapq.heappush(fringe, (total_estimated_cost[start_node], start_node))

    while fringe:
        current_node = heapq.heappop(fringe)[1]

        if current_node == end_node:
            path = []
            while current_node in parent:
                path.append(current_node)
                current_node = parent[current_node]
            path.append(fringe)
            return path[::-1]

        visited.add(current_node)

        for neighbor, cost in graph[current_node]:
            if neighbor in visited:
                continue

            tentative_actual_cost = actual_cost[current_node] + cost
            if neighbor not in actual_cost or tentative_actual_cost < actual_cost[neighbor]:
                parent[neighbor] = current_node
                actual_cost[neighbor] = tentative_actual_cost
                total_estimated_cost[neighbor] = tentative_actual_cost + \
                    Heuristic(neighbor, end_node, lat_long_dict)
                heapq.heappush(fringe, (total_estimated_cost[neighbor], neighbor))

    return None

if __name__=="__main__":
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
    breadthFirstSearch(a.graph)
    depthFirstSearch(a.graph)
    print(astar("Arad", "Bucharest", a.graph , Data))