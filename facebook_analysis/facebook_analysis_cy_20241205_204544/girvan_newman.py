import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import time

def load_saved_results(results_dir):
    """Load previously saved betweenness and community data"""
    results_dir = Path(results_dir)
    
    # Load betweenness centrality
    print("Loading betweenness centrality data...")
    with open(results_dir / 'betweenness_centrality.json', 'r') as f:
        bt_centrality = {int(k): float(v) for k, v in json.load(f).items()}
    
    # Load community data
    print("Loading community data...")
    with open(results_dir / 'communities.pickle', 'rb') as f:
        communities = pickle.load(f)
    
    return bt_centrality, communities

def analyze_community_structure(bt_centrality, communities):
    """Analyze relationships between betweenness and community structure"""
    
    # Calculate community-level metrics
    community_metrics = []
    for i, community in enumerate(communities):
        # Get betweenness values for community members
        community_bt = {node: bt_centrality.get(node, 0) for node in community}
        
        metrics = {
            'community_id': i,
            'size': len(community),
            'avg_betweenness': np.mean(list(community_bt.values())),
            'max_betweenness': max(community_bt.values()),
            'min_betweenness': min(community_bt.values()),
            'total_betweenness': sum(community_bt.values())
        }
        community_metrics.append(metrics)
    
    return community_metrics

def create_advanced_visualizations(community_metrics, output_dir):
    """Create detailed visualizations of community and betweenness relationships"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Community Size vs Average Betweenness
    plt.figure(figsize=(12, 8))
    sizes = [m['size'] for m in community_metrics]
    avg_bt = [m['avg_betweenness'] for m in community_metrics]
    
    plt.scatter(sizes, avg_bt, alpha=0.6)
    plt.xlabel('Community Size')
    plt.ylabel('Average Betweenness Centrality')
    plt.title('Community Size vs Average Betweenness')
    
    # Add trend line
    z = np.polyfit(sizes, avg_bt, 1)
    p = np.poly1d(z)
    plt.plot(sizes, p(sizes), "r--", alpha=0.8, label=f'Trend line')
    plt.legend()
    
    plt.savefig(output_dir / 'community_size_vs_betweenness.png')
    plt.close()
    
    # 2. Distribution of Betweenness within Communities
    plt.figure(figsize=(15, 8))
    # Select top 10 largest communities for visualization
    top_communities = sorted(community_metrics, key=lambda x: x['size'], reverse=True)[:10]
    
    data_to_plot = [
        [m['min_betweenness'], m['avg_betweenness'], m['max_betweenness']] 
        for m in top_communities
    ]
    
    plt.boxplot(data_to_plot, labels=[f"Comm {m['community_id']}" for m in top_communities])
    plt.title('Betweenness Distribution in Top 10 Largest Communities')
    plt.ylabel('Betweenness Centrality')
    plt.xlabel('Community')
    plt.xticks(rotation=45)
    
    plt.savefig(output_dir / 'community_betweenness_distribution.png')
    plt.close()
    
    # 3. Community Influence Distribution
    plt.figure(figsize=(12, 8))
    total_bt = [m['total_betweenness'] for m in community_metrics]
    
    # Calculate percentage of total influence
    total_influence = sum(total_bt)
    influence_percentages = [(bt/total_influence)*100 for bt in total_bt]
    
    # Sort communities by influence
    sorted_indices = np.argsort(influence_percentages)[::-1]
    top_20_percentages = [influence_percentages[i] for i in sorted_indices[:20]]
    top_20_ids = [community_metrics[i]['community_id'] for i in sorted_indices[:20]]
    
    plt.bar(range(len(top_20_percentages)), top_20_percentages)
    plt.title('Top 20 Communities by Network Influence')
    plt.xlabel('Community Rank')
    plt.ylabel('Percentage of Total Network Influence')
    
    # Add percentage labels
    for i, percentage in enumerate(top_20_percentages):
        plt.text(i, percentage, f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.savefig(output_dir / 'community_influence_distribution.png')
    plt.close()
    
    # Save numerical analysis
    with open(output_dir / 'community_analysis_detailed.txt', 'w') as f:
        f.write("Community Structure Analysis\n")
        f.write("==========================\n\n")
        
        # Overall statistics
        f.write("Overall Network Statistics:\n")
        f.write(f"Total Communities: {len(community_metrics)}\n")
        f.write(f"Average Community Size: {np.mean(sizes):.2f}\n")
        f.write(f"Median Community Size: {np.median(sizes):.2f}\n")
        f.write(f"Largest Community Size: {max(sizes)}\n")
        f.write(f"Smallest Community Size: {min(sizes)}\n\n")
        
        # Top communities by different metrics
        f.write("Top 10 Communities by Size:\n")
        for m in sorted(community_metrics, key=lambda x: x['size'], reverse=True)[:10]:
            f.write(f"Community {m['community_id']}: {m['size']} nodes, "
                   f"avg betweenness: {m['avg_betweenness']:.6f}\n")
        
        f.write("\nTop 10 Communities by Average Betweenness:\n")
        for m in sorted(community_metrics, key=lambda x: x['avg_betweenness'], reverse=True)[:10]:
            f.write(f"Community {m['community_id']}: avg betweenness: {m['avg_betweenness']:.6f}, "
                   f"size: {m['size']} nodes\n")

def analyze_saved_network_data(results_dir):
    """Analyze previously saved network results"""
    print("Starting analysis of saved network data...")
    start_time = time.time()
    
    # Load saved data
    bt_centrality, communities = load_saved_results(results_dir)
    
    # Create output directory for new analysis
    output_dir = Path(results_dir) / 'advanced_community_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print("Analyzing community structure...")
    community_metrics = analyze_community_structure(bt_centrality, communities)
    
    print("Creating visualizations...")
    create_advanced_visualizations(community_metrics, output_dir)
    
    print(f"\nAnalysis complete. Results saved in: {output_dir}")
    print(f"Total analysis time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_saved_data.py <results_directory>")
        print("Example: python analyze_saved_data.py network_analysis_cy_20240105_123456")
        sys.exit(1)
    
    analyze_saved_network_data(sys.argv[1])
