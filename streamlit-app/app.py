import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import tempfile
import os
import time
from tqdm import tqdm
import community as community_louvain
import gc  # For garbage collection

def setup_plotting_style():
    sns.set_theme()
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 300

@st.cache_data
def analyze_network(_G):
    """Analyze network using Louvain community detection"""
    partition = community_louvain.best_partition(_G)
    
    community_dict = {}
    for node, com_id in partition.items():
        if com_id not in community_dict:
            community_dict[com_id] = set()
        community_dict[com_id].add(node)
    
    communities = list(community_dict.values())
    del community_dict
    gc.collect()
    
    return partition, communities

def plot_network_sample(G, partition, sample_size=1000):
    """Create network visualization with a smaller sample"""
    community_nodes = {}
    for node, com in partition.items():
        if com not in community_nodes:
            community_nodes[com] = []
        community_nodes[com].append(node)
    
    sampled_nodes = []
    total_nodes = G.number_of_nodes()
    for com, nodes in community_nodes.items():
        com_size = len(nodes)
        n_samples = int((com_size / total_nodes) * sample_size)
        if n_samples > 0:
            sampled_nodes.extend(np.random.choice(nodes, min(n_samples, len(nodes)), replace=False))
    
    # Create subgraph of sampled nodes
    G_sample = G.subgraph(sampled_nodes)
    
    plt.figure(figsize=(15, 15))
    try:
        pos = nx.spring_layout(G_sample, k=2/np.sqrt(G_sample.number_of_nodes()))
    except ImportError:
        # Fallback to simpler layout if scipy is not available
        pos = nx.shell_layout(G_sample)
    
    # Draw network with community colors
    nx.draw_networkx_edges(G_sample, pos, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G_sample, pos,
                          node_color=[partition.get(node, 0) for node in G_sample.nodes()],
                          cmap=plt.cm.tab20,
                          node_size=100,
                          alpha=0.7)
    
    plt.title(f'Network Visualization (Sample of {len(sampled_nodes)} nodes)\nColored by Community')
    plt.axis('off')
    fig = plt.gcf()
    plt.close()
    return fig

def plot_community_sizes(communities):
    """Plot community size distribution"""
    sizes = [len(c) for c in communities]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(sizes, bins=min(30, len(sizes)), ax=ax)
    ax.set_title('Community Size Distribution')
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Count')
    
    if max(sizes) / min(sizes) > 100:  # Use log scale if there's large variance
        ax.set_xscale('log')
    
    ax.axvline(np.mean(sizes), color='r', linestyle='--', 
               label=f'Mean: {np.mean(sizes):.1f}')
    ax.axvline(np.median(sizes), color='g', linestyle='--', 
               label=f'Median: {np.median(sizes):.1f}')
    ax.legend()
    
    plt.close()
    return fig

def display_network_stats(G, communities):
    """Display network statistics"""
    sizes = [len(c) for c in communities]
    
    stats = {
        "Total Nodes": G.number_of_nodes(),
        "Total Edges": G.number_of_edges(),
        "Average Degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "Total Communities": len(communities),
        "Average Community Size": np.mean(sizes),
        "Median Community Size": np.median(sizes),
        "Largest Community": max(sizes),
    }
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    metrics = list(stats.items())
    third = len(metrics) // 3
    
    # Display metrics in three columns
    for (key, value), column in zip(metrics[:third], [col1] * third):
        if isinstance(value, float):
            column.metric(key, f"{value:.2f}")
        else:
            column.metric(key, f"{value:,}")
            
    for (key, value), column in zip(metrics[third:2*third], [col2] * third):
        if isinstance(value, float):
            column.metric(key, f"{value:.2f}")
        else:
            column.metric(key, f"{value:,}")
            
    for (key, value), column in zip(metrics[2*third:], [col3] * (len(metrics) - 2*third)):
        if isinstance(value, float):
            column.metric(key, f"{value:.2f}")
        else:
            column.metric(key, f"{value:,}")

def plot_top_communities(communities):
    """Plot top communities by size"""
    sizes = [(i, len(c)) for i, c in enumerate(communities)]
    sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 15 communities
    top_sizes = sizes[:15]
    total_nodes = sum(len(c) for c in communities)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    percentages = [(size/total_nodes)*100 for _, size in top_sizes]
    bars = ax.bar(range(len(percentages)), percentages)
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentages[i]:.1f}%\n({top_sizes[i][1]:,} nodes)',
                ha='center', va='bottom')
    
    ax.set_title('Top 15 Communities by Size')
    ax.set_xlabel('Community Rank')
    ax.set_ylabel('Percentage of Total Nodes')
    
    plt.close()
    return fig

def main():
    st.set_page_config(page_title="Network Community Analysis", layout="wide")
    st.title("Network Community Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload network data file (edge list format)", 
                                   type=['txt'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner('Loading network...'):
                # Read the network
                G = nx.read_edgelist(tmp_path, nodetype=int)
                st.success(f'Network loaded successfully! ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)')
                
            with st.spinner('Detecting communities...'):
                # Analyze the network
                partition, communities = analyze_network(G)
                st.success(f'Found {len(communities)} communities!')
            
            # Display network statistics
            display_network_stats(G, communities)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs([
                "Community Sizes", 
                "Top Communities",
                "Network Sample"
            ])
            
            with tab1:
                st.pyplot(plot_community_sizes(communities))
            
            with tab2:
                st.pyplot(plot_top_communities(communities))
            
            with tab3:
                with st.spinner('Generating network visualization...'):
                    sample_size = min(1000, G.number_of_nodes())
                    fig = plot_network_sample(G, partition, sample_size)
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # This will print the full traceback
        finally:
            # Cleanup temporary file
            os.unlink(tmp_path)
            
            # Clear memory
            if 'G' in locals():
                del G
            if 'partition' in locals():
                del partition
            if 'communities' in locals():
                del communities
            gc.collect()

if __name__ == "__main__":
    main()
