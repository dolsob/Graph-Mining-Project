import networkx as nx
import math
import random
import operator
import numpy as np
from sklearn.linear_model import LinearRegression

# Load information from a file into a networkx graph
def loadGraph(G,f_name):
    f = open(f_name,'r')
    for line in f:
        if line[0] == '#' or line[0] == '%':
            continue
        line = line.replace('\t',' ').replace('\n',' ').split(',')
        line = [x for x in line if not x == ' ']
        G.add_edge(line[0],line[1],weight=int(line[2])+10)
        G[line[0]][line[1]]['time'] = line[3]
        
        
# Calculates the Jaccard coefficent for al pairs of users and create an edge if the value is 
# greater than 0.5
def Jaccard(G):
    users = {}
    for v1 in G.nodes():
        for v2 in G.nodes():
            if not v1 == v2:
                n1 = list(G.neighbors(v1))
                n2 = list(G.neighbors(v2))
                if n1 == [] and n2 == []:
                    continue
                intersection = [v for v in n1 if v in n2]
                num = len(intersection)/(len(set(n1+n2)))
                if num > 0.5:
                    G.add_edge(v1,v2,weight=0.5*num)
                    if v1 in users.keys():
                        users[v1].append(v2)
                    else:
                        users[v1] = [v2]
                    if v2 in users.keys():
                        users[v2].append(v1)
                    else:
                        users[v2] = [v1]
    return users

# Calculate the prediction values for users and products
def calculateValues(users,G):
    common_neighbors = {}
    adamic_adar = {}
    preferential_attachment = {}
    triadic_closure = {}
    for v1 in users.keys():
        #n1 = list(G.neighbors(v1))
        n1 = users[v1]
        for v2 in users[v1]:
            n2 = list(G.neighbors(v2))
            for v3 in G.neighbors(v2):
                n3 = list(G.neighbors(v3))
                intersection = [v for v in n3 if v in n1]     
            
                # Calculate the common neighbors value
                common_neighbors[(v1,v3)] = len(intersection)
                
                # Calculate the Adamic-Adar value
                aa = 0
                for u in intersection:
                    num = len(list(G.neighbors(u)))
                    if num > 1:
                        aa += 1/math.log(num)
                adamic_adar[(v1,v3)] = aa
                
                # Calculate the preferential attachment value
                preferential_attachment[(v1,v3)] = len(n3)*len(n1)
                
                # Calculate the triadic closure value
                weights = 0
                data = G.get_edge_data(v2,v3,default=0)
                if not data == 0:
                    weights = data['weight']
                w = G.get_edge_data(v1,v2,default=0)
                if not w == 0:
                    w = w['weight']                
                triadic_closure[(v1,v3)] = w+weights
    return (common_neighbors,adamic_adar,preferential_attachment,triadic_closure)
                
# Remove the edges used for training
def removeTrainEdges(G,predictions):
    for e in predictions:
        if G.has_edge(e[0][0],e[0][1]):
            predictions.remove(e)    
    return predictions
                
# Calculate percentage of correct predictions
def accuracy(predictions, G):
    correct = 0
    for e in predictions:
        if G.has_edge(e[0][0],e[0][1]):
            correct += 1
    return correct/len(predictions)
        
# Predict the existence of edges in the graph
def predictEdges(G, G_full):
    # Keep only 25% of the original edges
    edges = sorted(G_full.edges(data=True), key=lambda t: t[2].get('time', 1))
    edges = edges[:len(edges)//4]
    for e in edges:
        G.add_edge(e[0],e[1],weight=e[2].get('weight',0))
            
    # Compute the Jaccard Coefficient for all pair of users
    users = Jaccard(G)
    
    # Calculate the prediction values
    common_neighbors,adamic_adar,preferential_attachment,triadic_closure = calculateValues(users,G)
    
    # Sort the prediction values in descending order
    common_neighbors = sorted(common_neighbors.items(), key=operator.itemgetter(1),reverse=True)
    adamic_adar = sorted(adamic_adar.items(), key=operator.itemgetter(1),reverse=True)
    preferential_attachment = sorted(preferential_attachment.items(), 
                                     key=operator.itemgetter(1),reverse=True)
    triadic_closure = sorted(triadic_closure.items(), key=operator.itemgetter(1),reverse=True)
    
    # Remove training edes
    common_neighbors = removeTrainEdges(G,common_neighbors)
    adamic_adar = removeTrainEdges(G,adamic_adar)
    preferential_attachment = removeTrainEdges(G,preferential_attachment)
    triadic_closure = removeTrainEdges(G,triadic_closure)
    
    # Keep only top 25% of predictions
    cn_predictions = common_neighbors[:len(common_neighbors)//4]
    aa_predictions = adamic_adar[:len(adamic_adar)//4]
    pa_predictions = preferential_attachment[:len(preferential_attachment)//4]
    tc_predictions = triadic_closure[:len(triadic_closure)//4]
    
    # Output accuracy of each prediction
    print("Common Neighbors Accuracy:", accuracy(cn_predictions,G_full))
    print("Adamic-Adar Accuracy:", accuracy(aa_predictions,G_full))
    print("Preferential Attachment Accuracy:", accuracy(pa_predictions,G_full))
    print("Triadic Closure Accuracy:", accuracy(tc_predictions,G_full))   
    
# Calculate the average of the weights of the incoming and outgoing edges for each vertex
def calculateAverages(G_full):
    in_edge_avg = {}
    out_edge_avg = {}
    for vertex in G_full.nodes:
        weights = 0
        c = 0
        for v1, v2, w in G_full.in_edges(vertex,data=True):
            weights += (w['weight'])
            c += 1
        if c > 0:
            in_edge_avg[vertex] = weights/c
            
        weights = 0
        c = 0
        for v1, v2, w in G_full.out_edges(vertex,data=True):
            weights += (w['weight'])
            c += 1
        if c > 0:
            out_edge_avg[vertex] = weights/c  
    return (in_edge_avg, out_edge_avg)
    
# Convert the averages into vectors
def calculatePoints(G_full, in_edge_avg, out_edge_avg):
    X = []
    Y = []
    for v1, v2, w in G_full.edges.data('weight'):
        if v1 in in_edge_avg and v1 in out_edge_avg and v2 in in_edge_avg and v2 in out_edge_avg:
            X.append((in_edge_avg[v1], out_edge_avg[v1], in_edge_avg[v2], out_edge_avg[v2]))
            Y.append(w)
    return X,Y
      
# Train the linear regression model  
def train(G, G_full):
    in_edge_avg, out_edge_avg = calculateAverages(G_full)
    X,Y = calculatePoints(G_full, in_edge_avg, out_edge_avg)
    reg = LinearRegression().fit(X, Y)
    return reg

# Test the linear regression model
def test(G_full, reg): 
    in_edge_avg, out_edge_avg = calculateAverages(G_full)
    X,Y = calculatePoints(G_full, in_edge_avg, out_edge_avg)
    MSE = 0
    for i in range(len(X)):
        p = reg.predict([X[i]])
        MSE += (p[0]-Y[i])**2
    return MSE/len(X)

if __name__ == '__main__':
    
    # Calculate the most reputable users
    f_name = 'soc-sign-bitcoinotc.csv'
    G = nx.DiGraph()
    G_full = nx.DiGraph()
    loadGraph(G_full, f_name)
    pr = nx.pagerank(G_full)
    pr = {k: v for k, v in sorted(pr.items(), key=lambda item: item[1],reverse=True)}
    
    t = 5
    print("Top",t,"Users")
    i = 1
    for key in pr.keys():
        print(i,": ",key,", ",pr[key],sep="")
        if i == 5:
            break
        else:
            i += 1
            
    print("Bottom",t,"Users")
    i = 1
    for key in pr.keys():
        if i > len(pr.keys())-5:
            print(i,": ",key,", ",pr[key],sep="")
        i += 1    

    
    # Predict edges in the graph
    print("\n","Edge Prediction",sep="")
    predictEdges(G,G_full)
    
    
    # Predict the weights of edges
    print("\n","Weight Prediction",sep="")
    reg = train(G, G_full)
    MSE = test(G_full, reg)
    print("Bitcoin Alpha Training MSE:",MSE)
    
    f_name = 'soc-sign-bitcoinalpha.csv'
    G = nx.DiGraph()
    G_full = nx.DiGraph()    
    loadGraph(G_full, f_name)
    MSE = test(G_full, reg)
    print("Bitcoin OTC Testing MSE:",MSE)  
    