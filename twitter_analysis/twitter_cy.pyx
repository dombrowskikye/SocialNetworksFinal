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
ctypedef int DTYPE_t
ctypedef double DTYPE_float_t

@cython.boundscheck(False)
@cython.wraparound(False)
def read_network_cy(str filename):
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
    
    return nx.DiGraph(edges)  # Using directed graph for Twitter

@cython.boundscheck(False)
@cython.wraparound(False)
def _single_source_shortest_path_basic(G, s):
    cdef:
        dict pred = dict({s: []})  # Convert to regular dict
        dict sigma = dict()        # Initialize as regular dict
        dict d = dict({s: 0})      # Convert to regular dict
        list queue = [s]
    
    sigma[s] = 1.0
    while queue:
        v = queue.pop(0)
        # Convert G[v] to regular dict if it's a defaultdict
        neighbors = dict(G[v]) if isinstance(G[v], defaultdict) else G[v]
        for w in neighbors:
            if w not in d:
                queue.append(w)
                d[w] = d[v] + 1
            if d[w] == d[v] + 1:
                if w not in sigma:
                    sigma[w] = 0.0
                sigma[w] += sigma[v]
                if w not in pred:
                    pred[w] = []
                pred[w].append(v)
    
    return pred, sigma, d

@cython.boundscheck(False)
@cython.wraparound(False)
def _accumulate_edges(G, s, dict pred, dict sigma, dict d):
    cdef:
        dict betweenness = dict()  # Initialize as regular dict
        dict delta = dict()        # Initialize as regular dict
        list stack
    
    # Convert to regular dict if needed
    pred = dict(pred)
    sigma = dict(sigma)
    d = dict(d)
    
    stack = sorted(pred.keys(), key=lambda x: -d[x])
    
    for node in pred:
        delta[node] = 0.0
    
    for w in stack:
        coefficient = (1.0 + delta[w]) / sigma[w]
        for v in pred[w]:
            edge = tuple(sorted([v, w]))
            c = sigma[v] * coefficient
            delta[v] += c
            if edge not in betweenness:
                betweenness[edge] = 0.0
            betweenness[edge] += c
    
    return betweenness

def process_chunk(args):
    G, nodes_chunk = args
    # Convert G to regular dict if needed
    if isinstance(G, nx.Graph):
        G = nx.Graph(dict(G.adj))
    elif isinstance(G, nx.DiGraph):
        G = nx.DiGraph(dict(G.adj))
    
    local_betweenness = {}
    for s in nodes_chunk:
        pred, sigma, d = _single_source_shortest_path_basic(G, s)
        betweenness = _accumulate_edges(G, s, pred, sigma, d)
        for edge, value in betweenness.items():
            if edge not in local_betweenness:
                local_betweenness[edge] = 0.0
            local_betweenness[edge] += value
    return local_betweenness

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_edge_betweenness_centrality(G, int num_workers=4):
    cdef:
        dict edge_betweenness = {}
        list nodes = list(G.nodes())
        int n_nodes = len(nodes)
        int chunk_size
    
    chunk_size = max(1, n_nodes // num_workers)
    node_chunks = [nodes[i:i + chunk_size] for i in range(0, n_nodes, chunk_size)]
    
    # Create argument tuples for process_chunk
    chunk_args = [(G, chunk) for chunk in node_chunks]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunk_args))
    
    for chunk_result in chunk_results:
        for edge, value in chunk_result.items():
            if edge not in edge_betweenness:
                edge_betweenness[edge] = 0.0
            edge_betweenness[edge] += value
    
    cdef float scale = 1.0 / (n_nodes * (n_nodes - 1))
    for edge in edge_betweenness:
        edge_betweenness[edge] *= scale
    
    return edge_betweenness

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_girvan_newman(G, max_communities=None, int num_workers=4):
    cdef:
        int n_communities = -1
    
    if max_communities is not None:
        n_communities = max_communities
    
    # Convert to undirected and ensure regular dict
    if not isinstance(G, nx.Graph):
        G = G.to_undirected()
    G = nx.Graph(dict(G.adj))
    
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
    
    pos = nx.spring_layout(G_undirected)
    
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
    
    G = read_network_cy(filename)
    print(f"Network loaded in {time.time() - start_time:.2f} seconds")
    
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_weakly_connected_components(G)  # Changed for directed graph
    }
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nCalculating betweenness centrality...")
    bt_start = time.time()
    bt_centrality = nx.betweenness_centrality(G)
    print(f"Betweenness centrality calculated in {time.time() - bt_start:.2f} seconds")
    
    with open(output_dir / 'betweenness_centrality.json', 'w') as f:
        json.dump({str(k): float(v) for k, v in bt_centrality.items()}, f)
    
    print("\nDetecting communities...")
    comm_start = time.time()
    if community_method == "girvan_newman":
        communities, removed_edges = parallel_girvan_newman(G, n_communities)
        with open(output_dir / 'removed_edges.pickle', 'wb') as f:
            pickle.dump(removed_edges, f)
    else:
        communities = list(nx.community.greedy_modularity_communities(G))
    
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