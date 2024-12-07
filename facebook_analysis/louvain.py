import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

def parallel_layout_calculation(nodes_batch, G, k=None, iterations=50):
    """Calculate spring layout for a batch of nodes in parallel"""
    pos = nx.spring_layout(G.subgraph(nodes_batch), k=k, iterations=iterations)
    return {node: coords for node, coords in pos.items()}

def efficient_spring_layout(G, n_jobs=None):
    """Parallel implementation of spring layout"""
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Split nodes into batches
    nodes = list(G.nodes())
    batch_size = max(len(nodes) // n_jobs, 1)
    node_batches = [nodes[i:i + batch_size] for i in range(0, len(nodes), batch_size)]
    
    # Calculate layout in parallel
    with Pool(n_jobs) as pool:
        partial_func = partial(parallel_layout_calculation, G=G)
        results = pool.map(partial_func, node_batches)
    
    # Combine results
    pos = {}
    for result in results:
        pos.update(result)
    
    return pos

def plot_louvain_communities(G, partition, output_file=None):
    """Plot communities with optimized rendering"""
    # Use efficient parallel layout
    print("Calculating layout...")
    pos = efficient_spring_layout(G)
    
    # Pre-calculate all visual elements
    print("Preparing visualization...")
    unique_communities = set(partition.values())
    colors = list(mcolors.CSS4_COLORS.values())[:len(unique_communities)]
    color_map = dict(zip(unique_communities, colors))
    
    # Batch node color assignment
    node_colors = np.array([color_map[partition[node]] for node in G.nodes()])
    
    # Efficient plotting
    plt.figure(figsize=(12, 12))
    
    # Draw with optimized parameters
    nx.draw(G, pos,
           node_color=node_colors,
           with_labels=False,
           node_size=50,
           edge_color='gray',
           alpha=0.7,
           linewidths=0,  # Remove node borders
           width=0.1)     # Thinner edges
    
    plt.title("Network Graph with Louvain Communities")
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def main():
    # Load the graph
    print("Loading graph...")
    file_path = 'data/facebook_combined.txt'
    G = nx.read_edgelist(file_path, nodetype=int)
    
    # Perform community detection
    print("Detecting communities...")
    partition = community_louvain.best_partition(G)
    
    # Plot the communities
    print("Plotting communities...")
    plot_louvain_communities(G, partition, output_file='communities.png')
    
    # Print some statistics
    print("\nNetwork Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of communities: {len(set(partition.values()))}")

if __name__ == "__main__":
    main()
