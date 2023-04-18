from Graph import Graph
import random
from Searches import *
import timeit
import matplotlib.pyplot as plt
import numpy as np


def generate_random_graph(n, p):
    graph = Graph()
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and random.random() < p:
                graph.insertEdge(i, j, 1)
    return graph

if __name__=="__main__":
    graph1 = generate_random_graph(10, 0.2)
    graph2 = generate_random_graph(10, 0.4)
    graph3 = generate_random_graph(10, 0.6)
    graph4 = generate_random_graph(10, 0.8)

    graph5 = generate_random_graph(20, 0.2)
    graph6 = generate_random_graph(20, 0.4)
    graph7 = generate_random_graph(20, 0.6)
    graph8 = generate_random_graph(20, 0.8)

    graph9 = generate_random_graph(30, 0.2)
    graph10 = generate_random_graph(30, 0.4)
    graph11 = generate_random_graph(30, 0.6)
    graph12 = generate_random_graph(30, 0.8)

    graph13 = generate_random_graph(40, 0.2)
    graph14 = generate_random_graph(40, 0.4)
    graph15 = generate_random_graph(40, 0.6)
    graph16 = generate_random_graph(40, 0.8)

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
    
    Graphs = [graph1, graph2, graph3, graph4, graph5, graph6, graph7, graph8, graph9, graph10, graph11, graph12, graph13, graph14, graph15, graph16]
    graph_paths = []
    graph_times = []
    for graph in Graphs:
        random_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        algos = [dfs1, bfs, ucs, bidirectional, greedy, astar, iterative_deepening_search]
        algo_path_length = []
        algo_time_taken = []
        for algo in algos:
            algo_path =0
            algo_time = 0
            for i in range(len(random_nodes)):
                for j in range(i, len(random_nodes)):
                    start_time = timeit.default_timer()
                    if algo == dfs1 or algo == bfs:
                        path = algo(random_nodes[i], random_nodes[j], graph.graph)
                    if algo == greedy:
                        path = greedy(graph.graph, random_nodes[i], random_nodes[j], graph.locations)
                    if algo == iterative_deepening_search:
                        path = iterative_deepening_search(graph.graph, random_nodes[i], random_nodes[j], 4)
                    if algo == astar:
                        path = astar(graph.graph, random_nodes[i], random_nodes[j], graph.locations)
                    if algo == ucs or algo == bidirectional:
                        path = algo(graph, random_nodes[i], random_nodes[j])
                    end_time = timeit.default_timer()
                
                if path:
                    algo_path += find_taken_distance(path, graph.graph)
                algo_time += end_time - start_time

            algo_path_length.append(algo_path)
            algo_time_taken.append(algo_time)

        graph_paths.append(algo_path_length)
        graph_times.append(algo_time_taken)


    def draw_plot(data, title, y_label):
        print(data)
        labels = ["dfs", "bfs", "ucs", "bidirectional", "greedy", "astar", "iterative"]

        x_values = list(range(1, 17))

        color_map = plt.get_cmap('Set1')
        num_colors = len(data[0])
        colors = [color_map(i/num_colors) for i in range(num_colors)]

        fig, ax = plt.subplots()

        for i in range(len(data)):
            ax.scatter([x_values[i]]*7, data[i], marker='o', color=colors, label=i, s=150)

        ax.set_xticks(x_values)
        ax.set_xticklabels(x_values)

        ax.set_xlabel('Graphs')
        ax.set_ylabel(y_label)

        ax.set_title(title)

        handles, _ = ax.get_legend_handles_labels()

        handles = [plt.Line2D([], [], marker='o', color=colors[i], label=labels[i]) for i in range(len(labels))]
        ax.legend(handles=handles)

        plt.show()
    
    draw_plot(graph_times, "Path analysis", "Path length")
    # draw_plot(graph_paths, "Path analysis", "Path length")
    