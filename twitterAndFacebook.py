import networkx as nx
import matplotlib.pyplot as plt


#Twitter file
twitter_file = "twitter_combined.txt"

#Facebook file
facebook_file = "facebook_combined.txt"



#Twitter Graph

twitter_graph = nx.read_edgelist(twitter_file, create_using=nx.DiGraph(), nodetype=int)
print("Twitter preproccessed")


#Facebook Graph

facebook_graph = nx.read_edgelist(facebook_file, create_using=nx.Graph(), nodetype=int)
print("Facebook preprocessed")


#Drawing twitter and facebook graphs
print("Graphing Twitter")
plt.figure(figsize=(30, 30))
nx.draw_spectral(twitter_graph, node_size=1, with_labels=False)
print("Twitter graphed.")
print("Saving Twitter graphs")
plt.savefig('twitter_graphs.png')
print("Graphs for Twitter saved. ")


print("Graphing Facebook")
plt.figure(figsize=(30, 30))
nx.draw_spectral(facebook_graph, node_size=1, with_labels=False)
print("Facebook graphed!")
print("Saving Facebook graphs")
plt.savefig('facebook_graphs.png')
print("Graphs for Facebook saved")



