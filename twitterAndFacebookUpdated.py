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
print("Computing Twitter")

N, K = twitter_graph.order(), twitter_graph.size()
twitter_avg_deg = float(K)/N
print("Nodes: ", N)
print("Edges: ", K) 
print("Average degree: ", twitter_avg_deg) 
#nx.draw_spectral(twitter_graph)
print("Twitter Computed.")

print("Computing Facebook")
NN, KK = facebook_graph.order(), facebook_graph.size()
facebook_avg_deg = float(KK)/NN
print("Nodes: ", NN)
print("Edges: ", KK)
print("Average degree: ", facebook_avg_deg) 
#nx.draw_spectral(facebook_graph)
print("Facebook computed!")


#print("Saving graphs")
#plt.savefig('twitter_and_facebook_graphs.png')
#print("Graphs saved")



