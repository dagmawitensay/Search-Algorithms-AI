from collections import deque
from Graph import Graph
import heapq
import math
import networkx as nx


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
    visited = {start, }
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
with open("./citys.txt", "r") as file:
    for line in file:
        line = line.split()
        if len(line) > 3:
            Data[f"{line[0]} {line[1]}"] = (line[2], line[3])
        else:
            Data[line[0]] = (line[1], line[2])
del Data['City']


def Heuristic(node_A, node_B, cityData):
    """"
    calculates the distance between the two cities in km using Haversine formula
    """
    long1, lat1, long2, lat2 = map(math.radians, [float(cityData[node_A][1]), float(
        cityData[node_A][0]), float(cityData[node_B][0]) * (math.pi / 180), float(cityData[node_B][0])])

    # Haversine formula
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371      # Radius of earth in kilometers.
    distance = c * r

    return distance


def astar(start_node, end_node, graph, lat_long_dict):
    fringe = []
    visited = set()
    parent = {}
    actual_cost = {start_node: 0}
    total_estimated_cost = {start_node: Heuristic(
        start_node, end_node, lat_long_dict)}

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
                heapq.heappush(
                    fringe, (total_estimated_cost[neighbor], neighbor))

    return None


def bfs(graph, starting_city, destination):
    # cities = a.graph.keys()
    visited = set()
    queue = deque([starting_city])
    pathlength = 1
    while queue:
        current = queue.popleft()
        visited.add(current)
        pathlength += 1
        if current == destination:
            return pathlength
        for neighbour in graph.getChilds(current):
            if neighbour not in visited:
                queue.append(neighbour)
            # queue = queue + (graph.getChilds(current))
    return visited


def dijkstra(start_vertex):
    path = {}
    adj_node = {}
    queue = []
    for node in a.graph:
        path[node] = float("inf")
        adj_node[node] = None
        queue.append(node)

    path[start_vertex] = 0
    while queue:
        key_min = queue[0]
        min_val = path[key_min]
        for n in range(1, len(queue)):
            if path[queue[n]] < min_val:
                key_min = queue[n]
                min_val = path[key_min]
        cur = key_min
        queue.remove(cur)

        for adjecent, cost in a.graph[cur]:
            alternate = cost + path[cur]
            if path[adjecent] > alternate:
                path[adjecent] = alternate
                adj_node[adjecent] = cur
    return adj_node


def shortest_paths(paths, parent, end, path=[]) -> None:
    if (end == -1):
        paths.append(path.copy())
        return
    for par in parent[end]:
        path.append(end)
        shortest_paths(paths, parent, par, path)
        path.pop()


def bfs(parent, start) -> None:

    dist = {node: float("inf") for node in a.graph}
    q = deque()
    q.append(start)
    parent[start] = [-1]
    dist[start] = 0
    while q:
        vertix = q.popleft()
        for neighbour, cost in a.graph[vertix]:
            if (dist[neighbour] > dist[vertix] + 1):
                dist[neighbour] = dist[vertix] + 1
                q.append(neighbour)
                parent[neighbour].clear()
                parent[neighbour].append(vertix)

            elif (dist[neighbour] == dist[vertix] + 1):
                parent[neighbour].append(vertix)


def print_shortest_paths(start, end) -> None:
    paths = []
    parent = {node: [] for node in a.graph}
    bfs(parent, start)
    shortest_paths(paths, parent, end)
    # print("from", start, "to", end)
    # print(paths)
    for v in paths:
        v = reversed(v)
        # for u in v:
        #     # print(u, end=" ")
        # print()
    return paths


def Closeness_Centralities(city):
    number_of_node = len(a.graph) - 1
    ans = dijkstra(city)
    denominator = 0
    for node in a.graph:
        # print(f'The path between {city} to {node}',end=" ")
        count = 0
        while True:
            node = ans[node]
            # print(node,end=" ")
            if node is None:
                # print(count,end=" ")
                denominator += count
                break
            count += 1
    # print(denominator)
    return number_of_node / denominator


def Betweenness_Centralities(city):
    keys = list(a.graph.keys())
    keys.remove(city)
    answer = 0
    top = 0
    bottom = 0
    for index in range(len(keys)):
        for ind in range(index+1, len(keys)):
            ans = print_shortest_paths(keys[index], keys[ind])
            # print(ans)
            times = 0
            for path in ans:
                if city in path:
                    times += 1
            top += times
            bottom += len(ans)
    return top/bottom



if __name__ == "__main__":

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

    # cities = list(a.graph.keys())
    # Closeness_Centralities_of_cities = {}
    # Betweenness_Centralities_of_cities = {}
    # for city in cities:
    #     Closeness_Centralities_of_cities[city] = Closeness_Centralities(city)
    #     Betweenness_Centralities_of_cities[city] = Betweenness_Centralities(city)
    # print("Closeness_Centralities \n\n", "    ", Closeness_Centralities_of_cities )
    # print("\n\n *************************************************************")
    # print("************************************************************* \n\n ")
    # print("Betweeness_Centralities \n\n", "    ", Betweenness_Centralities_of_cities)
    # print( " \n\n  The answer below  gives the answer in array and the above answer gives in dictionary \n\n")

    # print(list(Betweenness_Centralities_of_cities.values()))

