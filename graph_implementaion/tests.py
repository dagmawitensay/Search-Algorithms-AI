from Graph import Graph
import random
from Searches import astar, Data
import timeit

def generate_random_graph(n, p):
        graph = Graph()
        for i in range(1, n + 1):
            graph.locations[i] = (random.uniform(1.0, 100.0) , random.uniform(1.0, 100.0))
            for j in range(1, n + 1):
                if random.random() < p:
                    graph.insertEdge(i, j, random.randrange(1, 150))
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

        
