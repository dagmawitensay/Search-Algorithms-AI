from collections import deque
import math
from collections import defaultdict, deque
import random
import time
import networkx as nx


def haversine_distance(x1, y1, x2, y2):
    lat1_rad = math.radians(x1)
    lon1_rad = math.radians(y1)
    lat2_rad = math.radians(x2)
    lon2_rad = math.radians(y2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * \
        math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 6371 * c
    return distance


coordinates = {

    'Arad': (46.1866, 21.3123),
    'Zerind': (46.6225, 21.5174),
    'Oradea': (47.0465, 21.9189),
    'Timisoara': (45.7489, 21.2087),
    'Lugoj': (45.6910, 21.9035),
    'Mehadia': (44.9041, 22.3645),
    'Drobeta': (44.6369, 22.6597),
    'Craiova': (44.3302, 23.7949),
    'Rimnicu Vilcea': (45.0997, 24.3693),
    'Sibiu': (45.7936, 24.1213),
    'Fagaras': (45.8416, 24.9731),
    'Pitesti': (44.8565, 24.8692),
    'Giurgiu': (43.9037, 25.9699),
    'Bucharest': (44.4268, 26.1025),
    'Urziceni': (44.7181, 26.6453),
    'Eforie': (44.0491, 28.6527),
    'Hirsova': (44.6893, 27.9457),
    'Vaslui': (46.6407, 27.7276),
    'Iasi': (47.1585, 27.6014),
    'Neamt': (46.9759, 26.3819)

}


def heuristic_values(goal_location):

    goal_latitude, goal_longitude = coordinates[goal_location]
    heuristic_values = {}
    for location, (latitude, longitude) in coordinates.items():
        heuristic_value = haversine_distance(
            latitude, longitude, goal_latitude, goal_longitude)
        heuristic_values[location] = heuristic_value
    return heuristic_values


goal_location = 'Bucharest'
heuristic_values = heuristic_values(goal_location)
print("Heuristic Values:")
for location, heuristic_value in heuristic_values.items():
    print(location, heuristic_value)


def greedy_search(graph, start_location, goal_location, heuristic_values):

    if start_location == goal_location:
        return [start_location]
    visited = set()
    frontier = deque([[start_location]])
    while frontier:
        path = frontier.popleft()
        current_location = path[-1]
        if current_location not in visited:
            visited.add(current_location)
            if current_location == goal_location:
                return path
            neighbors = graph[current_location]
            # Sort neighbors based on heuristic values
            neighbors = sorted(neighbors, key=lambda x: heuristic_values[x])
            for neighbor in neighbors:
                if neighbor not in visited:
                    frontier.append(path + [neighbor])
    return None


graph = {
    'Arad': ['Zerind', 'Sibiu', 'Timisoara'],
    'Zerind': ['Oradea', 'Arad'],
    'Oradea': ['Zerind', 'Sibiu'],
    'Timisoara': ['Arad', 'Lugoj'],
    'Lugoj': ['Timisoara', 'Mehadia'],
    'Mehadia': ['Lugoj', 'Drobeta'],
    'Drobeta': ['Mehadia', 'Craiova'],
    'Craiova': ['Drobeta', 'Rimnicu Vilcea', 'Pitesti'],
    'Rimnicu Vilcea': ['Craiova', 'Sibiu', 'Pitesti'],
    'Sibiu': ['Oradea', 'Arad', 'Fagaras', 'Rimnicu Vilcea'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Pitesti': ['Craiova', 'Rimnicu Vilcea', 'Bucharest'],
    'Bucharest': ['Fagaras', 'Pitesti', 'Urziceni', 'Giurgiu'],
    'Urziceni': ['Bucharest', 'Hirsova', 'Vaslui'],
    'Eforie': ['Hirsova'],
    'Hirsova': ['Urziceni', 'Eforie'],
    'Vaslui': ['Urziceni', 'Iasi'],
    'Iasi': ['Vaslui', 'Neamt'],
    'Neamt': ['Iasi'],
    'Giurgiu': ['Bucharest']
}


# 2 a
start_location = 'Arad'
goal_location = 'Rimnicu Vilcea'
num_runs = 10
total_time_taken = 0

for i in range(num_runs):
    start_time = time.time()
    result = greedy_search(graph, start_location,
                           goal_location, heuristic_values)
    end_time = time.time()
    time_taken = end_time - start_time
    total_time_taken += time_taken

average_time_taken = total_time_taken / num_runs
print("Average time taken:", average_time_taken, "seconds")


# 2 b

# Define your greedy search function
def greedy_search(graph, start_location, goal_location, heuristic_values):
    # code for the greedy
    visited_nodes = set()
    current_location = start_location
    solution_length = 0

    while current_location != goal_location:
        visited_nodes.add(current_location)
        next_location = None
        min_heuristic_value = float('inf')

        for neighbor in graph[current_location]:
            if neighbor not in visited_nodes:
                heuristic_value = heuristic_values[neighbor]
                if heuristic_value < min_heuristic_value:
                    min_heuristic_value = heuristic_value
                    next_location = neighbor

        if next_location is None:
            return None

        current_location = next_location
        solution_length += 1
    return solution_length


solution_length = greedy_search(
    graph, start_location, goal_location, heuristic_values)
if solution_length is None:
    print("No solution found.")
else:
    print("Solution length:", solution_length)


# 2 i)


# Define the number of nodes and the probability of edges
# num_nodes = [10, 20, 30, 40]
# prob_edges = [0.2, 0.4, 0.6, 0.8]

#


# def generate_random_graph(num_nodes, prob_edges):
#     graph = {}
#     nodes = [f'Node{i}' for i in range(1, num_nodes+1)]
#     for node in nodes:
#         graph[node] = []
#         for neighbor in nodes:
#             if node != neighbor and random.random() < prob_edges:
#                 graph[node].append(neighbor)
#     return graph


# # Generate random graphs
# random_graphs = {}
# for n in num_nodes:
#     for p in prob_edges:
#         graph_name = f'n_{n}_p_{p}'
#         random_graph = generate_random_graph(n, p)
#         random_graphs[graph_name] = random_graph

# # Function to randomly select five nodes


# def select_five_nodes(graph):
#     return random.sample(list(graph.keys()), 5)


# # Loop through the random graphs and apply the haversine_distance() algorithm
# for graph_name, graph in random_graphs.items():
#     print(f'Graph: {graph_name}')
#     nodes = select_five_nodes(graph)
#     for node in nodes:
#         print(f'Start Node: {node}')
#         for location, (latitude, longitude) in coordinates.items():
#             heuristic_value = heuristic_values[location]
#             print(
#                 f'Goal Location: {location}, Heuristic Value: {heuristic_value}')


# # Generate random graphs and find paths
# for n in [10, 20, 30, 40]:
#     for p in [0.2, 0.4, 0.6, 0.8]:
#         print(f"Number of Nodes (n): {n}, Edge Probability (p): {p}")
#         for i in range(5):
#             graph = nx.erdos_renyi_graph(n, p, seed=i, directed=False)
#             coordinates = {node: (random.uniform(0, 90), random.uniform(
#                 0, 180)) for node in graph.nodes()}
#             start_location = random.choice(list(graph.nodes()))
#             goal_location = random.choice(list(graph.nodes()))
#             while start_location == goal_location:
#                 goal_location = random.choice(list(graph.nodes()))
#             heuristic_values = {}
#             for location, (latitude, longitude) in coordinates.items():
#                 heuristic_value = haversine_distance(
#                     latitude, longitude, coordinates[goal_location][0], coordinates[goal_location][1])
#                 heuristic_values[location] = heuristic_value
#             print(
#                 f"Start Location: {start_location}, Goal Location: {goal_location}")
#             path = greedy_search(graph, start_location,
#                                  goal_location, heuristic_values)
#             print(f"Path: {path}")
#             print("---")


def generate_random_graph(n, p):
    graph = {}
    nodes = list(range(n))
    for i in range(n):
        neighbors = []
        for j in range(n):
            if i != j and random.random() < p:
                neighbors.append(j)
        graph[i] = neighbors
    return graph


num_nodes = [10, 20, 30, 40]
edge_probabilities = [0.2, 0.4, 0.6, 0.8]

num_nodes_selected = 5
nodes_selected = random.sample(range(max(num_nodes)), num_nodes_selected)

for n in num_nodes:
    for p in edge_probabilities:
        graph = generate_random_graph(n, p)
        print(f"Random graph with n = {n} nodes and p = {p} edge probability:")
        print(graph)
        print("Selected nodes:", nodes_selected)
        coordinates = {location: (random.uniform(0, 90), random.uniform(
            0, 180)) for location in graph.keys()}
        for start_location in nodes_selected:
            for goal_location in nodes_selected:
                if start_location != goal_location:
                    heuristic_values = heuristic_values(
                        goal_location, coordinates)
                    path = greedy_search(
                        graph, start_location, goal_location, heuristic_values)
                    print(
                        f"Start: {start_location}, Goal: {goal_location}, Path: {path}")
        print("------------------------")
