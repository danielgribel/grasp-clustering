#!/usr/bin/env python

# summary: grasp algorithm for clustering.
# based on "Investigation of a new GRASP-based clustering algorithm applied to biological data" paper

# paper author: Maria C.V. Nascimento, Franklina M.B. Toledo, Andre C.P.L.F. de Carvalho
# code author: Daniel Lemes Gribel (dgribel@inf.puc-rio.br)

import numpy
import random
import networkx as nx
import time
from numpy.linalg import norm
from scipy.spatial import distance
from heapq import nlargest, nsmallest

# reading input data
def load_data(inputdata):
	mydata = numpy.loadtxt(inputdata, dtype = numpy.object, delimiter = ',')
	return mydata

def draw_graph(G):
	# an example using Graph as a weighted network.
	# __author__ = """Aric Hagberg (hagberg@lanl.gov)"""
	try:
	    import matplotlib.pyplot as plt
	except:
	    raise

	elarge = [(u,v) for (u,v,d) in G.edges(data = True) if d['weight'] > 0.5]
	esmall = [(u,v) for (u,v,d) in G.edges(data = True) if d['weight'] <= 0.5]

	pos = nx.spring_layout(G) # positions for all nodes

	# nodes
	nx.draw_networkx_nodes(G, pos, node_size = 200)

	# edges
	nx.draw_networkx_edges(G, pos, edgelist = elarge, width = 0.4)
	nx.draw_networkx_edges(G, pos, edgelist = esmall, width = 0.4, alpha = 0.6, style = 'dashed')

	# labels
	nx.draw_networkx_labels(G, pos, font_size = 6, font_family = 'sans-serif')

	print 'number of cliques/clusters:', nx.graph_number_of_cliques(G)
	print 'time:', time.time() - start
	plt.show()

	#filename = "w_graph" + ".png"
	#plt.savefig(filename)

def constructive_phase(weights, G, N, M):
	for k in range(0, M-1):
		S = dict()
		if N < 15:
			# let L be a list with every edge of the graph
			L = weights
		else:
			n_h = min(0.1 * N * (N-1)/2, len(G.edges()))
			# L = list with the n_h highest weighted edges
			L = nlargest(int(n_h), weights)

		cmin = min(L)
		cmax = max(L)

		alpha = random.random()
		RCL = list()

		for e in G.edges():
			if G[e[0]][e[1]]['weight'] >= cmax + alpha*(cmin-cmax) and G[e[0]][e[1]]['weight'] <= cmax:
				RCL.append(e)

		# choose randomly an element from RCL. This edge connects a node i to another node j inside the graph.
		r = random.randint(0, len(RCL)-1)
		i = RCL[r][0]
		j = RCL[r][1]
		
		intersec = set(G.neighbors(i)).intersection(G.neighbors(j))
		
		# let S_i and S_j be two empty sets of nodes.
		S[i] = set()
		S[j] = set()
		S[i].add(i)
		S[j].add(j)

		# consider all elements v such that (v, i) in G and (v, j) in G, i != j.
		for v in intersec:
			if G[v][i]['weight'] < G[v][j]['weight']:
				S[i].add(v)
			else:
				S[j].add(v)

		weights = list()

		# eliminate from the graph G the edges (v_t , v_z) in V where v_t in S_i and v_z in S_j
		for e in G.edges():
			if (e[0] in S[i] and e[1] in S[j]) or (e[0] in S[j] and e[1] in S[i]):
				G.remove_edge(e[0], e[1])
			else:
				weights.append(G[e[0]][e[1]]['weight'])

		#for i1 in range(0, len(intersec)):
		#	for i2 in range(i1+1, len(intersec)):
		#		v1 = list(intersec)[i1]
		#		v2 = list(intersec)[i2]
		#		if ((v1 in S[i]) and (v2 in S[j])) or ((v2 in S[i]) and (v1 in S[j])):
		#			G.remove_edge(v1, v2)
		#		else:
		#			weights.append(G[v1][v2]['weight'])

	print list(nx.find_cliques(G))
	draw_graph(G)

def demo():
	mydata = load_data('iris.csv')

	N = len(mydata) # number of instances
	M = 3 # number of clusters
		
	# removing last field: label/classification
	mydata_att = numpy.delete(mydata, 4, 1)

	# converting matrix (originally string) to float
	mydata_att = mydata_att.astype(numpy.float)

	# initializing distance matrix/graph
	G = nx.Graph()
	weights = list()
		
	# calculating distance matrix: euclidean distance
	for i in range(0, N):
		for j in range(i+1, N):
			w = distance.euclidean(mydata_att[i], mydata_att[j])
			G.add_edge(i, j, weight = w)
			weights.append(w)
			
	constructive_phase(weights, G, N, M)

if __name__ == "__main__":
	start = time.time()
	demo()