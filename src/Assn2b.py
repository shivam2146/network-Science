import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
from pprint import pprint

def draw_graph(G, title):
    pos = nx.spring_layout(G)
    nx.draw(G_er, pos, with_labels = True )
    nx.draw_networkx_labels(G,pos,font_size=9)
    plt.title('Network - ER Model')
    plt.show()
    
'''
Calculating and plotting the degree dist
'''
def degree_dist(G):
    deg = G.degree()
    #print(deg)
    #print(type(deg))
    deg_dict = dict(deg)
    #print(deg_dict)
    degree_list = deg_dict.values()
    #print(degree_list)
    #Counter makes a multiset
    degree_dic = Counter(degree_list)
    print('Degree Distribution: ')
    pprint(degree_dic)
    prob_d = [i/150 for i in degree_dic.values()]
    degree_hist = pd.DataFrame({'Degree':list(degree_dic.keys()),'Probability':prob_d})
    print(degree_hist)
    
    #bar plot
    sns.barplot(y = 'Probability', x = 'Degree', data = degree_hist)
    plt.title('Degree Frequency Distribution Plot')
    plt.xlabel('Degree')
    plt.ylabel('Probability') 
    #plt.show()
    
    #line plot
    sns.lineplot(y = 'Probability', x = 'Degree', data = degree_hist)
    plt.title('Degree Frequency Distribution Plot')
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.show()
    
'''
Calculating and plotting clustering coefficient dist
'''
def clus_co(G):
    clus_coeff = nx.clustering(G)
    #print(clus_coeff)
    clus_list = [round(v,2) for v in clus_coeff.values()]
    #print(clus_list)
    clus_dict = Counter(clus_list)
    #print(clus_dict)
    clus_coeff_hist = pd.DataFrame({'Frequency':list(clus_dict.values()),'Clustering coefficient':list(clus_dict.keys())})
    sns.barplot(x = 'Clustering coefficient', y = 'Frequency', data = clus_coeff_hist)
    #plt.hist(clus_coeff_hist)
    plt.title('Clustering coefficient Plot')
    plt.xlabel('Clustering coefficient')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    #plt.show()
    
    
    #line plot
    sns.lineplot(y = 'Frequency', x = 'Clustering coefficient', data = clus_coeff_hist)
    plt.title('Clustering coefficient Plot')
    plt.xlabel('Clustering coefficient')
    plt.ylabel('Frequency')
    plt.show()
    
'''
Small world property
'''
def small_world_prop(G):
    try: 
        print(nx.average_shortest_path_length(G))
    except NetworkXError:
        print("Graph is not connected")
        for g in nx.connected_component_subgraphs(G):
            print(nx.average_shortest_path_length(g))

#Generating network using ER model
G_er = nx.erdos_renyi_graph(100,0.2)
draw_graph(G_er,'Network - ER Model')

#Generating network using WS model
G_ws = nx.watts_strogatz_graph(100,2,0.2)
draw_graph(G_ws,'Network - WS Model' )

#Generating network using BA model
G_ba = nx.barabasi_albert_graph(100,3)
draw_graph(G_ba,'Network - BA Model')

#Analysing real world properties

#Degree dist
degree_dist(G_er)
degree_dist(G_ws)
degree_dist(G_ba)

#Clus coeff
clus_co(G_er)
clus_co(G_ws)    
clus_co(G_ba)

#small_world
small_world_prop(G_er)
small_world_prop(G_ws)
small_world_prop(G_ba)
    
#Checking for assortativity and disassortativity
#print(nx.degree_assortativity_coefficient(G_er))
#print(nx.degree_assortativity_coefficient(G_ws))
#print(nx.degree_assortativity_coefficient(G_ba))