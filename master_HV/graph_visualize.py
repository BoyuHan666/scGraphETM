import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


# Example adjacency matrix
adj_matrix = np.array([[0, 1, 1],
                       [1, 0, 0],
                       [0, 1, 0]])

# Create a new graph
G = nx.Graph()

# Add nodes
rows, columns = adj_matrix.shape
for i in range(rows):
    G.add_node(f"Row {i+1}")
    G.add_node(f"Column {i+1}")

# Add edges
# progress_bar = tqdm(total=np.sum(adj_matrix))
for i in tqdm(range(rows), desc='Adding edges', unit='All Peak per gene'):
    for j in range(columns):
        if adj_matrix[i, j] == 1:
            G.add_edge(f"Row {i+1}", f"Column {j+1}")

# Draw the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, node_color="skyblue", font_color="black")
plt.title("Matrix as a Graph")
plt.savefig('./plot_July24/' + 'graph_visualize' + '.png')
