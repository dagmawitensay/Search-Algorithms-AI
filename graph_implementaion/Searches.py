from collections import deque
from Graph import Graph
import heapq
import math
import random
import util

def dfs1(start, end, adj_list, path = []):
    path = path + [start]
    if start == end:
        return path
    for node in adj_list[start]:
        if node[0] not in path:
            new_path = dfs1(node[0], end, adj_list, path)
            if new_path:
                return new_path
    return None


def bfs(start, end, adj_list):
    queue = deque()
    visited = set()

    queue.append((start, [start]))

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path

        visited.add(node)
        for neighbor, cost in adj_list[node]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None

def ucs(graph,source,destination):
    parent = {}
    closed = []
    open = util.PriorityQueue()
    open.push(source,0)
    
    while True:
        
        if open.isEmpty():
            break
        
        else:
            selected_node = open.pop()
            
        if selected_node == destination:
            break

        
        if selected_node not in closed:
            closed.append(selected_node)
    
        children = graph.neighbours(selected_node)

        if children != None:
            for child in children:
                city,cost = child
                if city not in parent.keys():
                    parent[city] = selected_node
        
                if city == destination:
                    closed.append(city)
                    break
        
                if cost != None and city not in closed:
                    open.push(city,cost)
    path = [destination]
    curr = destination
    while curr != source:
        curr = parent[curr]
        path.insert(0,curr)

    return path


def bidirectional(graph,source,destination):
    source_parent = {}
    source_visited = []
    source_queue = util.Queue()
    dest_parent = {}
    dest_visited = []
    path = []
    dest_queue = util.Queue()

    source_queue.push(source)
    dest_queue.push(destination)
    
    if graph.neighbours(destination):
        for ele in graph.neighbours(destination):
            if ele[0] == source:
                return [source, destination]


    while True and source_queue:
        selected = source_queue.pop()
        source_visited.append(selected)
        connected = graph.neighbours(selected)

        if connected != None:
            for child in connected:
                city,cost = child

                if city not in source_parent.keys() and city not in source_parent:
                    source_parent[city] = selected
            
                if city not in source_queue.list and city not in source_visited:
                    source_queue.push(city)
        

        selected_dest = dest_queue.pop()
        dest_visited.append(selected_dest)
        connected_dest = graph.neighbours(selected_dest)
        

        if connected_dest != None:
            for child in connected_dest:
                city_dest, cost_dest = child
                
                if city_dest not in dest_parent.keys() and city_dest not in dest_parent.values():
                    dest_parent[city_dest] = selected_dest
                if city_dest not in dest_queue.list and city_dest not in dest_visited:
                    dest_queue.push(city_dest)

        
    
        for each in source_queue.list:
            if each == source or each == destination:
                    
                    break
            if each in dest_queue.list:
                path_dest = []
                current_dest = dest_parent[each]
                while current_dest != destination:
                    path_dest.append(current_dest)
    
                    current_dest = dest_parent[current_dest]
                path_dest.append(destination)
                
                current = each
            
                while current != source:
                    if current not in path:
                        path.insert(0,current)
                        current = source_parent[current]
                
                path.insert(0,source)
            
                return path + path_dest

def Heuristic(node_A, node_B, cityData):
    """"
    calculates the distance between the two cities in km using Haversine formula
    """
    long1, lat1, long2, lat2 = map(math.radians, [float(cityData[node_A][1]), float(cityData[node_A][0]),float(cityData[node_B][0]) * (math.pi / 180), float(cityData[node_B][0])])

    dlon = long2 - long1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371      # Radius of earth in kilometers.
    distance = c * r

    return distance

def find_taken_distance(path, graph):
    final_cost = 0
    visited = set()
    for i in range(len(path)):
        if i + 1 < len(path):
            for node in path:
                for neigbor in graph[node]:
                    if path[i + 1] == neigbor[0] and path[i + 1] not in visited:
                        visited.add(neigbor[0])
                        final_cost += neigbor[1]
    return final_cost

def iterative_deepening_search(graph, start, goal, max_depth):
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
        path = dfs(graph, start, goal, depth, visited)
        if path is not None:
            return path
    return None

def dfs(graph, node, goal, depth, visited):
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
    for neighbor, cost in graph[node]:
        if neighbor not in visited:
            path = dfs(graph, neighbor, goal, depth - 1, visited)
            if path is not None:
                return [node] + path

    return None

def greedy(adj_list, start, end, locations):
    """A* search algorithm using the Haversine formula as the heuristic function."""
    actual_cost = {start: 0} 
    total_estimated_cost = {start: Heuristic(start, end, locations)} 
    heap = [(total_estimated_cost[start], start)]
    visited = set()
    predecessor = {}
    
    while heap:
        _, current = heapq.heappop(heap)
        if current == end:
            path = []
            while current in predecessor:
                path.append(current)
                current = predecessor[current]
            path.append(start)
            path.reverse()
            return path
        
        visited.add(current)
        for neighbor, distance in adj_list[current]:
            if neighbor in visited:
                continue
            tentative_actual_cost = actual_cost[current] + distance
            if neighbor not in actual_cost or tentative_actual_cost < actual_cost[neighbor]:
                predecessor[neighbor] = current
                actual_cost[neighbor] = tentative_actual_cost
                total_estimated_cost[neighbor] = Heuristic(start, end, locations)
                heapq.heappush(heap, (total_estimated_cost[neighbor], neighbor))
    
    return None

def astar(adj_list, start, end, locations):
    """A* search algorithm using the Haversine formula as the heuristic function."""
    actual_cost = {start: 0} 
    total_estimated_cost = {start: Heuristic(start, end, locations)} 
    heap = [(total_estimated_cost[start], start)]
    visited = set()
    predecessor = {}
    
    while heap:
        _, current = heapq.heappop(heap)
        if current == end:
            path = []
            while current in predecessor:
                path.append(current)
                current = predecessor[current]
            path.append(start)
            path.reverse()
            return path
        
        visited.add(current)
        for neighbor, distance in adj_list[current]:
            if neighbor in visited:
                continue
            tentative_actual_cost = actual_cost[current] + distance
            if neighbor not in actual_cost or tentative_actual_cost < actual_cost[neighbor]:
                predecessor[neighbor] = current
                actual_cost[neighbor] = tentative_actual_cost
                total_estimated_cost[neighbor] = tentative_actual_cost + Heuristic(start, end, locations)
                heapq.heappush(heap, (total_estimated_cost[neighbor], neighbor))
    
    return None

Data = {}
with open("./citys.txt","r") as file:
    for line in file:
        line = line.split()
        if len(line) > 3:
            Data[f"{line[0]} {line[1]}"] = (line[2], line[3])
        else:
            Data[line[0]] = (line[1],line[2])
del Data['City']

cities = list(Data.keys())
randomCities = []
while len(randomCities) < 11:
    randomCities.append(cities[random.randint(0, 19)])

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
    a.locations = Data
    
    # path = dfs1("Arad", "Bucharest", a.graph)
    # path2 = bfs("Arad", "Bucharest", a.graph)
    # path3 = astar(a.graph, "Arad", "Bucharest", a.locations)
    path4 = ucs(a, "Arad", "Bucharest")
    path5 = bidirectional(a, "Arad", "Bucharest")
    # print(path, len(path))
    # print(find_taken_distance(path, a.graph))
    # print(path2, len(path2))
    # print(find_taken_distance(path2, a.graph))
    # print(path3, len(path3))
    # print(find_taken_distance(path3, a.graph))
    print(path4, len(path4))
    print(find_taken_distance(path4, a.graph))
    print(path5, len(path5))
    print(find_taken_distance(path5, a.graph))