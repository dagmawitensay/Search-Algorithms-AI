# import networkx as nx
# import matplotlib.pyplot as plt
# from Graph import Graph
# from Searches import Data

# a = Graph()
# a.insertEdge('Oradea','Sibiu', 151 )
# a.insertEdge('Oradea','Zerind', 71)
# a.insertEdge('Zerind','Arad', 75)
# a.insertEdge('Arad','Sibiu', 140)
# a.insertEdge('Arad','Timisoara', 118)
# a.insertEdge('Sibiu','Fagaras', 99)
# a.insertEdge('Sibiu','Rimnicu Vilcea', 80)
# a.insertEdge('Timisoara','Lugoj', 111)
# a.insertEdge('Lugoj','Mehadia', 70)
# a.insertEdge('Mehadia','Drobeta', 75)
# a.insertEdge('Drobeta','Craiova', 120)
# a.insertEdge('Craiova','Pitesti', 138)
# a.insertEdge('Rimnicu Vilcea','Pitesti', 97)
# a.insertEdge('Rimnicu Vilcea', 'Craiova', 146)
# a.insertEdge('Fagaras', 'Bucharest', 211)
# a.insertEdge('Pitesti', 'Bucharest', 101)
# a.insertEdge('Bucharest', 'Urziceni', 85)
# a.insertEdge('Bucharest', 'Giurgiu', 90)
# a.insertEdge('Urziceni', 'Hirsova', 98)
# a.insertEdge('Urziceni', 'Vaslui', 142)
# a.insertEdge('Hirsova', 'Eforie', 86)
# a.insertEdge('Vaslui', 'Iasi', 92)
# a.insertEdge('Iasi', 'Neamt', 87)
# a.locations = Data

# G = nx.Graph()

# for node, neighbors in a.graph.items():
#     G.add_node(node)
#     for neighbor, cost in neighbors:
#         G.add_edge(node, neighbor, weight=cost)

# pos = nx.spring_layout(G)

# nx.draw_networkx_nodes(G, pos, node_size=1000)
# nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
# nx.draw_networkx_labels(G, pos)

# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Example data (16 lists, each containing 5 elements)
data = [[1, 2, 3, 4, 5],
        [2, 4, 6, 70, 10],
        [3, 6, 9, 12, 15],
        [4, 8, 12, 16, 20],
        [5, 10, 30, 20, 25],
        [6, 12, 18, 24, 30],
        [7, 14, 21, 28, 35],
        [8, 16, 24, 32, 40],
        [9, 18, 27, 36, 45],
        [10, 55, 30, 40, 50],
        [11, 22, 33, 44, 55],
        [12, 24, 36, 48, 60],
        [13, 26, 39, 52, 65],
        [14, 78, 42, 56, 70],
        [15, 30, 45, 60, 75],
        [16, 32, 48, 64, 80]]

def draw_plot(data, title, y_label):
    # Labels for each index position
    labels = ["dfs", "bfs", "greedy", "astar", "iterative"]

    # Create a list of x values (from 1 to 16)
    x_values = list(range(1, 17))

    # Create a color map with a unique color for each index position
    color_map = plt.get_cmap('Set1')
    num_colors = len(data[0])
    colors = [color_map(i/num_colors) for i in range(num_colors)]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Loop through each list in the data and plot as a scatter plot with unique color
    for i in range(len(data)):
        ax.scatter([x_values[i]]*5, data[i], marker='o', color=colors, label=i)

    # Set the x-axis tick labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values)

    # Set the x and y axis labels
    ax.set_xlabel('Graphs')
    ax.set_ylabel(y_label)

    # Set the plot title
    ax.set_title(title)

    # Create a legend with the labels
    handles, _ = ax.get_legend_handles_labels()

    handles = [plt.Line2D([], [], marker='o', color=colors[i], label=labels[i]) for i in range(len(labels))]
    # labels_legend = [labels[i] for i in range(len(data[0]))]
    ax.legend(handles=handles)

    # Show the plot
    plt.show()

draw_plot(data, "Time analysis", "time taken")
