import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_plotting_style():
    sns.set_theme()
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 300

def plot_betweenness_distribution(bt_centrality, output_dir):
    """Plot distribution of betweenness centrality values"""
    plt.figure(figsize=(12, 8))
    values = list(bt_centrality.values())
    
    # Create histogram
    sns.histplot(values, bins=50)
    plt.title('Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Count')
    plt.yscale('log')  # Log scale for better visualization
    
    # Add statistical annotations
    plt.axvline(np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.6f}')
    plt.axvline(np.median(values), color='g', linestyle='--', label=f'Median: {np.median(values):.6f}')
    plt.legend()
    
    plt.savefig(output_dir / 'betweenness_distribution_detailed.png')
    plt.close()

    # Create a top nodes analysis
    top_nodes = sorted(bt_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    plt.figure(figsize=(15, 8))
    node_ids, centrality_values = zip(*top_nodes)
    
    sns.barplot(x=range(len(node_ids)), y=centrality_values)
    plt.title('Top 20 Nodes by Betweenness Centrality')
    plt.xlabel('Node Rank')
    plt.ylabel('Betweenness Centrality')
    plt.xticks(range(len(node_ids)), [f"Node {id}" for id in node_ids], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_betweenness_nodes.png')
    plt.close()

def plot_community_analysis(communities, output_dir):
    """Create detailed community analysis plots"""
    # Community size distribution
    sizes = [len(c) for c in communities]
    
    plt.figure(figsize=(12, 8))
    sns.histplot(sizes, bins=30)
    plt.title('Community Size Distribution')
    plt.xlabel('Community Size')
    plt.ylabel('Count')
    plt.yscale('log')
    
    # Add statistics
    plt.axvline(np.mean(sizes), color='r', linestyle='--', label=f'Mean: {np.mean(sizes):.1f}')
    plt.axvline(np.median(sizes), color='g', linestyle='--', label=f'Median: {np.median(sizes):.1f}')
    plt.legend()
    
    plt.savefig(output_dir / 'community_size_distribution.png')
    plt.close()

    # Top communities by size
    plt.figure(figsize=(15, 8))
    top_n = 20
    top_sizes = sorted(sizes, reverse=True)[:top_n]
    
    sns.barplot(x=range(top_n), y=top_sizes)
    plt.title(f'Top {top_n} Largest Communities')
    plt.xlabel('Community Rank')
    plt.ylabel('Number of Nodes')
    
    # Add percentage labels
    total_nodes = sum(sizes)
    for i, size in enumerate(top_sizes):
        percentage = (size / total_nodes) * 100
        plt.text(i, size, f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_communities.png')
    plt.close()

def create_community_summary(communities, output_dir):
    """Create a text summary of community statistics"""
    sizes = [len(c) for c in communities]
    
    summary = {
        'Total Communities': len(communities),
        'Total Nodes': sum(sizes),
        'Average Community Size': np.mean(sizes),
        'Median Community Size': np.median(sizes),
        'Largest Community': max(sizes),
        'Smallest Community': min(sizes),
        'Size Standard Deviation': np.std(sizes)
    }
    
    # Calculate size percentiles
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        summary[f'{p}th Percentile Size'] = np.percentile(sizes, p)
    
    # Write summary to file
    with open(output_dir / 'community_analysis.txt', 'w') as f:
        f.write("Community Analysis Summary\n")
        f.write("=========================\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.2f}\n")

def analyze_saved_results(results_dir):
    """Analyze previously saved results and create new visualizations"""
    results_dir = Path(results_dir)
    output_dir = results_dir / 'additional_analysis'
    output_dir.mkdir(exist_ok=True)
    
    setup_plotting_style()
    
    # Load betweenness centrality data
    print("Loading betweenness centrality data...")
    with open(results_dir / 'betweenness_centrality.json', 'r') as f:
        bt_centrality = {int(k): float(v) for k, v in json.load(f).items()}
    
    # Load community data
    print("Loading community data...")
    with open(results_dir / 'communities.pickle', 'rb') as f:
        communities = pickle.load(f)
    
    # Create visualizations
    print("Generating betweenness centrality visualizations...")
    plot_betweenness_distribution(bt_centrality, output_dir)
    
    print("Generating community analysis visualizations...")
    plot_community_analysis(communities, output_dir)
    
    print("Creating community summary...")
    create_community_summary(communities, output_dir)
    
    print(f"\nAnalysis complete. Additional visualizations saved in: {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python visualize_results.py <results_directory>")
        print("Example: python visualize_results.py network_analysis_cy_20240105_123456")
        sys.exit(1)
    
    analyze_saved_results(sys.argv[1])
