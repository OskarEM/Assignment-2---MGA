# Genetic Algorithm for Graph Coloring

This project implements a **Genetic Algorithm (GA)** to solve the **Graph Coloring Problem**. The task involves coloring a graph using two colors (black and white) such that no two connected vertices share the same color. The problem is solved using genetic algorithms with various evolutionary techniques like selection, crossover, and mutation.

## Table of Contents
- [Project Overview](#project-overview)
- [Theoretical Background](#theoretical-background)
  - [Genetic Algorithm](#genetic-algorithm)
  - [The Coloring Problem](#the-coloring-problem)
- [Implementation](#implementation)
  - [Principal Steps](#principal-steps)
  - [Code Explanation](#code-explanation)
- [Results](#results)
  - [Optimization and Generations](#optimization-and-generations)
- [Discussion](#discussion)
  - [Filling Unmapped Vertices](#filling-unmapped-vertices)
  - [Generations Required for Optimization](#generations-required-for-optimization)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

In this project, a **Genetic Algorithm** is applied to solve the **Graph Coloring Problem** using two colors: black and white. The goal is to assign colors to the vertices such that no two adjacent vertices share the same color. The process starts by reducing the problem size progressively, from a 16-vertex graph down to 4 vertices, solving the smaller problems and scaling the solution back up to the original size.

## Theoretical Background

### Genetic Algorithm
Genetic algorithms are inspired by natural evolution, utilizing mechanisms like selection, crossover, and mutation to solve optimization problems. These algorithms operate on a population of potential solutions (chromosomes) and evolve over multiple generations to find an optimized solution.

### The Coloring Problem
The graph coloring problem involves assigning colors to vertices of a graph so that no two adjacent vertices share the same color. In this project, the goal is to minimize the number of conflicts (edges connecting vertices of the same color) using a GA-based approach.

## Implementation

### Principal Steps
1. **Graph Initialization**: A 16-vertex graph is created, either using a circular pattern of edges or randomly generated edge patterns.
2. **Problem Reduction**: The graph is progressively reduced in size from 16 vertices to 8 vertices, and finally to 4 vertices.
3. **Genetic Algorithm**: The genetic algorithm is applied to solve the 4-vertex problem, and the solution is expanded back to the 16-vertex graph.

### Code Explanation
The main code includes:
- `Graph Class`: Represents the graph with its vertices and edges.
- `GeneticAlgorithm Class`: Implements the genetic algorithm functions, including:
  - **Initial Population Generation**
  - **Fitness Function**
  - **Selection**: Selects the fittest individuals.
  - **Crossover**: Two-point crossover to generate offspring.
  - **Mutation**: Randomly flips genes (colors).
  - **Graph Reduction and Expansion**: Reduces the graph and maps solutions back to the original graph size.

## Results

### Optimization and Generations
The optimization results showed that a larger number of generations leads to better solutions. Initially, 2 generations were used to demonstrate the algorithm, but it was found that using 100 generations yielded a more optimal solution.

- **Initial 4-Vertex Problem**: Fitness score of 2 after 2 generations.
- **8-Vertex Problem**: Fitness score of 7 after 2 generations.
- **16-Vertex Problem**: Fitness score of 14 after 2 generations.
- **Final 100 Generations**: The algorithm reached the global maximum with a fitness score of 16.

## Discussion

### Filling Unmapped Vertices
Different strategies could have been used to fill unmapped vertices during problem expansion:
- **Copy Solution**
- **Two-Way Solution**
- **Random Filling with Proportional Distribution**
- **Nearest Neighbor Pattern**

In this project, we chose the simplest approach: filling the unmapped vertices with black.

### Generations Required for Optimization
It was observed that 50 to 100 generations were typically enough to find the global maxima. More generations yielded progressively better solutions.

## Conclusion

This project demonstrates how genetic algorithms can effectively solve the **Graph Coloring Problem** by reducing large graphs into smaller, more manageable problems, solving them, and scaling the solution back up. The use of crossover and mutation in the genetic algorithm ensures that optimal solutions are found efficiently.

