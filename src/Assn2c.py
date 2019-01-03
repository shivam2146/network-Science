import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import seaborn as sns
import networkx as nx
from collections import Counter

'''
Reading csv file into data frame
'''
iris = pd.read_csv('iris.csv')

'''
Constructing box plots for each feature Class wise
'''
plt.title('Feature box plot - Class wise')
sns.boxplot( x=iris["Class"], y=iris["Sepal Length"])
plt.show()
plt.title('Feature box plot - Class wise')
sns.boxplot( x=iris["Class"], y=iris["Sepal Width"])
plt.show()
plt.title('Feature box plot - Class wise')
sns.boxplot( x=iris["Class"], y=iris["Petal Length"])
plt.show()
plt.title('Feature box plot - Class wise')
sns.boxplot( x=iris["Class"], y=iris["Petal Width"])
plt.show()


y = iris['Class'] 
x = iris.drop('Class', axis = 1)

'''
Calculating eucledian distances between data points
'''
diff = []
for i in range(149):
    for j in range(i+1,150):
        d = 0
        for c in range(4):
            d += ((x.iloc[i,c] - x.iloc[j,c])**2)
        diff.append(math.sqrt(d))
        
distance_matrix = np.asarray(diff)

'''
Constructing box plot of the distances
'''
plt.ylabel('Distance')
plt.title('Pairwise Distance - Box plot')
box = plt.boxplot(distance_matrix,showmeans = True,showfliers=True)
plt.show()

'''
Calculating mean and std of the distances
'''
mean = np.mean(distance_matrix)
std = np.std(distance_matrix)

'''
Calculating z score of all the distances
'''
distance_matrix = (distance_matrix - mean)/std

'''
Taking threshhold as the 1st quartile value of distances
'''
#median = np.median(distance_matrix)
threshold = np.percentile(distance_matrix,25)
print('1st quartile uesd as threshold: ',threshold)

'''
Linking nodes(making edges) between data points whose dist <= threshold
'''
edge_list = []
k = 0
for i in range(149):
    for j in range(i+1,150):
        if(distance_matrix[k] <= threshold):
            edge_list.append((i,j))
        k = k+1
    #print(k)

#print('Edge list: ',edge_list)

'''
Constructing undirected graph using edge list
'''
G = nx.Graph()
node_list = range(150)
G.add_nodes_from(node_list)
G.add_edges_from(edge_list)
#print(G.nodes())
#print(G.edges())

print(nx.info(G))

'''
Drawing the graph
'''
node_color = "red"
edge_colors = range(0,13980,5)
nx.draw(G, pos=nx.spring_layout(G), node_color=node_color , edge_color= edge_colors, node_size=10,width=0.5)
plt.show()

'''
Calculating and plotting the degree dist
'''
deg = G.degree()
#print(deg)
#print(type(deg))
deg_dict = dict(deg)
#print(deg_dict)
degree_list = deg_dict.values()
#print(degree_list)
#Counter makes a multiset
degree_dic = Counter(degree_list)
print('Degree Distribution: ',degree_dic)
prob_d = [i/150 for i in degree_dic.values()]
degree_hist = pd.DataFrame({'Degree':list(degree_dic.keys()),'Probability':prob_d})
print(degree_hist)
sns.barplot(y = 'Probability', x = 'Degree', data = degree_hist)
plt.title('Degree Frequency Distribution Plot')
plt.xlabel('Degree')
plt.ylabel('Probability')
plt.show()

'''
Calculating and plotting clustering coefficient of the nodes
'''
clus_coeff = nx.clustering(G)
#print(clus_coeff)
clus_list = [round(v,2) for v in clus_coeff.values()]
#print(clus_list)
clus_dict = Counter(clus_list)
#print(clus_dict)
clus_coeff_hist = pd.DataFrame({'Frequency':list(clus_dict.values()),'Clustering coefficient':list(clus_dict.keys())})
sns.barplot(x = 'Clustering coefficient', y = 'Frequency', data = clus_coeff_hist)
plt.title('Clustering coefficient Plot')
plt.xlabel('Clustering coefficient')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.show()

'''
Calculating closeness centrality
'''
clos_cen = nx.closeness_centrality(G)

'''
Calculating betweeness centrality
'''
bet_cen = nx.betweenness_centrality(G)

'''
Calculating correlation btw these two centralities
'''
cent = pd.DataFrame({'Closeness centrality':list(clos_cen.values()),'Betweenness centrality':list(bet_cen.values())})
cor = cent['Closeness centrality'].corr(cent['Betweenness centrality'])
print('Correlation between centralities: ',cor)