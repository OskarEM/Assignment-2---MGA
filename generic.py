import multiprocessing
import random
import itertools
from multiprocessing import Pool


from collections import deque
import matplotlib.pyplot as plt
import networkx as nx




def visualize_graph(graph, solution):
    G = nx.Graph()
    G.add_edges_from(graph.get_edges())

    # Add nodes explicitly based on the solution list
    G.add_nodes_from(range(1, len(solution) + 1))

    # Check if there's an extra node and remove it
    if len(G.nodes()) > len(solution):
        # Identify the extra node. Assuming it's the highest-numbered node
        extra_node = max(G.nodes())
        # Remove the extra node
        G.remove_node(extra_node)

    # Adjust color_map creation to account for the solution list
    color_map = ['black' if solution[node - 1] == 'B' else 'white' for node in G.nodes()]

    # Draw the graph
    nx.draw(G, with_labels=True, node_color=color_map, node_size=700, font_weight='bold', font_color='red')
    plt.show()


class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        if u != v and u < self.num_vertices and v < self.num_vertices:
            self.edges[u].add(v)
            self.edges[v].add(u)

    def get_edges(self):
        return [(u, v) for u in range(self.num_vertices) for v in self.edges[u] if u < v]




class GeneticAlgorithm:
    def __init__(self, graph, population_size=4):
        self.graph = graph
        self.population_size = population_size
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [[random.choice(['B', 'W']) for _ in range(self.graph.num_vertices)] for _ in range(self.population_size)]

    def fitness(self, individual):

        return sum(1 for u, v in self.graph.get_edges() if individual[u] != individual[v])

    def select(self):
        return sorted(self.population, key=self.fitness, reverse=True)[:2]

    def two_point_crossover(self, parent1, parent2):
        point1, point2 = sorted(random.sample(range(self.graph.num_vertices), 2))
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2

    def mutate(self, individual, mutation_rate=0.1):
        return [gene if random.random() > mutation_rate else 'B' if gene == 'W' else 'W' for gene in individual]


    def run(self, generations=10, initial_population=None):
        if initial_population is not None:
            # Ensuring that the initial population matches the graph size
            self.population[0] = initial_population[:self.graph.num_vertices]
            for i in range(1, self.population_size):
                self.population[i] = self.population[i][:self.graph.num_vertices]

        for generation in range(generations):
            #print(f"Generation {generation + 1}:")
            # Show the current population's solutions and their fitness scores
            #for i, individual in enumerate(self.population):
            #    print(f"  Solution {i + 1}: {individual}, Fitness: {self.fitness(individual)}")
            # Selection
            selected = self.select()
            #print("  Selected parents for crossover:", selected)
            # Generate offspring
            offspring = []
            for _ in range(self.population_size // 2):
                child1, child2 = self.two_point_crossover(*selected)
                #print(f" Children of gen {generation + 1}: {child1}: {child2} ")
                # Mutation
                child1 = self.mutate(child1, 0.01)[:self.graph.num_vertices]
                child2 = self.mutate(child2, 0.01)[:self.graph.num_vertices]
                offspring.append(child1)
                offspring.append(child2)
            # Updating the population with offspring
            self.population = sorted(offspring, key=self.fitness, reverse=True)
            best_solution = self.population[0]
            #print(f"  Best Solution of Generation {generation + 1}: {best_solution}, Fitness: {self.fitness(best_solution)}\n")
        return self.population[0]



def reduce_graph(graph, target_vertices):
    new_graph = Graph(target_vertices)
    # Randomly choose a starting vertex
    start_vertex = random.choice(list(range(graph.num_vertices)))
    visited = set([start_vertex])  # Initialize visited with the start vertex
    queue = deque([start_vertex])

    # BFS to collect connected vertices up to target_vertices
    while queue and len(visited) < target_vertices:
        current_vertex = queue.popleft()
        neighbors = list(graph.edges[current_vertex])  # Get neighbors, adjust based on your Graph class implementation
        random.shuffle(neighbors)  # Shuffle neighbors to ensure randomness in selection

        for neighbor in neighbors:
            if neighbor not in visited:  # Check if the neighbor has not been visited
                visited.add(neighbor)  # Mark the neighbor as visited
                queue.append(neighbor)  # Add to queue for BFS
                if len(visited) == target_vertices:  # Stop if we reach the desired number of vertices
                    break

    # Build the new reduced graph
    old_vertex_indices = list(visited)
    old_to_new_indices = {old_index: i for i, old_index in enumerate(old_vertex_indices)}

    for u in old_vertex_indices:
        for v in graph.edges[u]:
            if v in old_vertex_indices:  # Ensure both vertices are in the selected subset
                new_u = old_to_new_indices[u]
                new_v = old_to_new_indices[v]
                new_graph.add_edge(new_u, new_v)  # Add edge to the new graph

    return new_graph, old_vertex_indices



def expand_solution(reduced_solution, old_vertex_indices, num_vertices):
    full_solution = ['B'] * num_vertices
    for new_index, color in enumerate(reduced_solution):
        old_index = old_vertex_indices[new_index]
        full_solution[old_index] = color
    return full_solution

def main():
    circle = True
    # Create the initial 16-vertex graph and add edges
    graph_16 = Graph(16)


    if not circle:
        # Add edges -
        for i in range(16):
            for j in range(i+1, 16):
                if random.random() < 0.3:  # Randomly create edges with 30% probability
                    graph_16.add_edge(i, j)

    if circle:
        graph_16 = Graph(16)

        for i in range(16):
            graph_16.add_edge(i, (i + 1) % 16)


    # Reduce graph from 16 to 8 vertices
    graph_8, indices_16_to_8 = reduce_graph(graph_16, 8)
    print(graph_8.edges)

    # Further reduce graph from 8 to 4 vertices
    graph_4, indices_8_to_4 = reduce_graph(graph_8, 4)
    print(graph_4.edges)

    # Apply GA to 4-vertex graph
    ga_4 = GeneticAlgorithm(graph_4)

    print(ga_4.population)
    solution_4 = ga_4.run(generations=100)
    print("4-vertex solution:", solution_4)

    # Expand solution from 4 to 8 vertices
    solution_8_initial = expand_solution(solution_4, indices_8_to_4, 8)

    # Apply GA to 8-vertex graph with initial solution
    ga_8 = GeneticAlgorithm(graph_8)
    solution_8 = ga_8.run(generations=100, initial_population=solution_8_initial)
    print("8-vertex solution:", solution_8)

    # Expand solution from 8 to 16 vertices
    solution_16_initial = expand_solution(solution_8, indices_16_to_8, 16)

    # Apply GA to 16-vertex graph with initial solution
    ga_16 = GeneticAlgorithm(graph_16)
    solution_16 = ga_16.run(generations=100, initial_population=solution_16_initial)
    print("16-vertex solution:", solution_16)
    visualize_graph(graph_16, solution_16)


if __name__ == '__main__':
    main()
