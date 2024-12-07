# twitter_cy.pyx
# cython: language_level=3
import cython
from cython.parallel import prange, parallel
import networkx as nx
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
import json
import pickle
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# Type declarations
ctypedef np.npy_int DTYPE_t
ctypedef np.npy_float64 DTYPE_float_t

@cython.boundscheck(False)
@cython.wraparound(False)
def read_twitter_network_cy(str filename):
    cdef:
        list edges = []
        int follower, followed
    
    with open(filename, 'r') as f:
        for line in tqdm(f, desc="Loading network"):
            if line.strip():
                try:
                    follower, followed = map(int, line.strip().split())
                    edges.append((follower, followed))
                except ValueError:
                    continue
    
    return nx.DiGraph(edges)

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_edge_betweenness_centrality(G, int num_workers=4):
    cdef:
        dict edge_betweenness = defaultdict(float)
        list nodes = list(G.nodes())
        int n_nodes = len(nodes)
        int chunk_size
    
    chunk_size = max(1, n_nodes // num_workers)
    node_chunks = [nodes[i:i + chunk_size] for i in range(0, n_nodes, chunk_size)]
    
    def process_chunk(nodes_chunk):
        chunk_betweenness = defaultdict(float)
        for s in nodes_chunk:
            pred, sigma, d = _single_source_shortest_path_basic(G, s)
            betweenness = _accumulate_edges(G, s, pred, sigma, d)
            for edge, value in betweenness.items():
                chunk_betweenness[edge] += value
        return dict(chunk_betweenness)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_chunk, node_chunks))
    
    for chunk_result in chunk_results:
        for edge, value in chunk_result.items():
            edge_betweenness[edge] += value
    
    cdef float scale = 1.0 / (n_nodes * (n_nodes - 1))
    for edge in edge_betweenness:
        edge_betweenness[edge] *= scale
    
    return dict(edge_betweenness)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _single_source_shortest_path_basic(G, s):
    cdef:
        dict pred = {s: []}
        dict sigma = defaultdict(float)
        dict d = {s: 0}
        list queue = [s]
        int v, w
    
    sigma[s] = 1.0
    while queue:
        v = queue.pop(0)
        for w in G[v]:
            if w not in d:
                queue.append(w)
                d[w] = d[v] + 1
            if d[w] == d[v] + 1:
                sigma[w] += sigma[v]
                if w not in pred:
                    pred[w] = []
                pred[w].append(v)
    
    return pred, dict(sigma), d

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _accumulate_edges(G, s, dict pred, dict sigma, dict d):
    cdef:
        dict betweenness = defaultdict(float)
        dict delta = defaultdict(float)
        list stack = sorted(pred.keys(), key=lambda x: -d[x])
    
    for w in stack:
        coefficient = (1.0 + delta[w]) / sigma[w]
        for v in pred[w]:
            edge = tuple(sorted([v, w]))
            c = sigma[v] * coefficient
            delta[v] += c
            betweenness[edge] += c
    
    return dict(betweenness)

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_girvan_newman(G, max_communities=None, int num_workers=4):
    cdef:
        int n_communities = -1
    
    if max_communities is not None:
        n_communities = max_communities
        
    if not isinstance(G, nx.Graph):
        G = G.to_undirected()
    
    g = G.copy()
    removed_edges = []
    
    with tqdm(total=g.number_of_edges(), desc="Detecting communities") as pbar:
        while g.number_of_edges() > 0:
            edge_betweenness = parallel_edge_betweenness_centrality(g, num_workers)
            max_edge = max(edge_betweenness.items(), key=lambda x: x[1])[0]
            g.remove_edge(*max_edge)
            removed_edges.append(max_edge)
            pbar.update(1)
            
            communities = list(nx.connected_components(g))
            if n_communities > 0 and len(communities) >= n_communities:
                break
    
    return communities, removed_edges

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_betweenness_sample_cy(G, int k, bint normalized=True, int seed=-1):
    cdef:
        dict betweenness
        list nodes, pivots
        int n, i
        float norm
        dict paths
    
    betweenness = {node: 0.0 for node in G}
    nodes = list(G.nodes())
    n = len(nodes)
    
    if seed >= 0:
        np.random.seed(seed)
    
    pivots = list(np.random.choice(nodes, min(k, n), replace=False))
    
    for pivot in tqdm(pivots, desc="Computing betweenness"):
        paths = dict(nx.single_source_shortest_path_length(G, pivot))
        for node in paths:
            if node != pivot:
                betweenness[node] += 1.0 / len(pivots)
    
    if normalized and n > 2:
        norm = 1.0 / ((n - 1) * (n - 2))
        for node in betweenness:
            betweenness[node] *= norm
    
    return betweenness

@cython.boundscheck(False)
@cython.wraparound(False)
def efficient_layout_cy(G, float k=-1.0, int iterations=50, int seed=42):
    cdef:
        int n = G.number_of_nodes()
        dict pos, displacement
        float temperature = 0.1
        int i, j
        float distance, length
        np.ndarray delta
    
    if k < 0:
        k = 1.0 / sqrt(n)
    
    np.random.seed(seed)
    pos = {node: np.random.rand(2) for node in G}
    
    for i in range(iterations):
        displacement = {node: np.zeros(2) for node in G}
        
        # Calculate forces
        for node1 in G:
            for node2 in G:
                if node1 != node2:
                    delta = pos[node1] - pos[node2]
                    distance = max(0.01, np.sqrt(np.sum(delta ** 2)))
                    displacement[node1] += k * k / distance * delta
        
        # Update positions
        for node in G:
            length = max(0.01, np.sqrt(np.sum(displacement[node] ** 2)))
            pos[node] += displacement[node] / length * min(length, temperature)
        
        temperature *= 0.95
    
    return pos

def create_output_directory():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"network_analysis_cy_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def plot_network_visualization_cy(G, dict bt_centrality, list communities, output_dir):
    cdef int TOP_NODES = 1000
    
    print("Preparing network visualization...")
    top_nodes = sorted(bt_centrality.items(), key=lambda x: x[1], reverse=True)[:TOP_NODES]
    top_node_ids = [node for node, _ in top_nodes]
    
    G_sub = G.subgraph(top_node_ids)
    G_undirected = G_sub.to_undirected()
    
    plt.figure(figsize=(15, 15))
    
    try:
        pos = efficient_layout_cy(G_undirected, iterations=50)
    except Exception as e:
        print(f"Layout calculation failed: {e}")
        return
    
    node_sizes = [bt_centrality.get(node, 0) * 5000 + 100 for node in G_sub.nodes()]
    community_colors = {}
    for i, community in enumerate(communities):
        for node in community:
            if node in top_node_ids:
                community_colors[node] = i
    
    node_colors = [community_colors.get(node, 0) for node in G_sub.nodes()]
    
    nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G_sub, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          cmap=plt.cm.tab20,
                          alpha=0.7)
    
    plt.title(f'Network Visualization (Top {TOP_NODES} nodes)\nNode size: Betweenness Centrality, Color: Community')
    plt.axis('off')
    plt.savefig(output_dir / 'network_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

def analyze_network_cy(str filename, str community_method="girvan_newman", max_communities=None):
    cdef:
        float start_time = time.time()
        dict bt_centrality
        list communities
        int n_communities = -1
    
    if max_communities is not None:
        n_communities = max_communities
    
    print("Starting Cython-optimized analysis...")
    output_dir = create_output_directory()
    
    G = read_twitter_network_cy(filename)
    print(f"Network loaded in {time.time() - start_time:.2f} seconds")
    
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_weakly_connected_components(G)
    }
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nCalculating approximate betweenness centrality...")
    bt_start = time.time()
    bt_centrality = calculate_betweenness_sample_cy(G, k=100, normalized=True, seed=42)
    print(f"Betweenness centrality calculated in {time.time() - bt_start:.2f} seconds")
    
    with open(output_dir / 'betweenness_centrality.json', 'w') as f:
        json.dump({str(k): float(v) for k, v in bt_centrality.items()}, f)
    
    print("\nDetecting communities...")
    comm_start = time.time()
    if community_method == "girvan_newman":
        communities, removed_edges = parallel_girvan_newman(G.to_undirected(), n_communities)
        # Save removed edges for potential reconstruction
        with open(output_dir / 'removed_edges.pickle', 'wb') as f:
            pickle.dump(removed_edges, f)
    else:
        communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
    
    print(f"Communities detected in {time.time() - comm_start:.2f} seconds")
    print(f"Number of communities found: {len(communities)}")
    
    with open(output_dir / 'communities.pickle', 'wb') as f:
        pickle.dump(communities, f)
    
    print("\nGenerating visualizations...")
    try:
        plot_network_visualization_cy(G, bt_centrality, communities, output_dir)
        print("Network visualization created")
    except Exception as e:
        print(f"Warning: Visualization failed due to: {str(e)}")
    
    print(f"\nAnalysis complete. Results saved in: {output_dir}")
    print(f"Total analysis time: {time.time() - start_time:.2f} seconds")