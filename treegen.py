import random
import numpy as np
#import networkx as nx
#import matplotlib.pyplot as plt
import copy
import math
import datetime
#import sys

def vertex(i):
  return chr(ord('A')+i)

def index(vertex):
  return ord(vertex)-ord('A')

def ispower(base,result):
  i=0
  while base**i < result:
    i+=1
  return base**i == result

def to_tree(input,rootnode):
   
   if ispower(2,len(input)/2+1) == False:
     print("Error")
     return 0
   lefttree,backbone,righttree=input[:int(len(input)/2)-1],input[int(len(input)/2)-1:int(len(input)/2)+1],input[int(len(input)/2)+1:]
   if backbone[0] == rootnode:
       othernode=backbone[1]
       backbone=backbone[1]+rootnode
   else:
       othernode=backbone[0]  
 
   if len(input) == 6:
    if lefttree[0] == rootnode:
        lefttree=lefttree[1]+rootnode
    if righttree[0] == othernode:
        righttree=righttree[1]+othernode
    return lefttree+","+righttree+","+backbone
   else:
    return to_tree(lefttree,rootnode=rootnode)+","+to_tree(righttree,rootnode=othernode)+","+backbone
   
edges_dict={
'AB':9,
'AC':3,
'AE':4,
'AJ':3,
'BC':9,
'BF':8,
'BK':4,
'CD':2,
'CE':4,
'CF':9,
'CM':4,
'DE':2,
'DF':8,
'DG':9,
'DO':5,
'DP':4,
'EG':4,
'EN':3,
'FG':7,
'FH':9,
'FO':5,
'GH':5,
'GP':5,
'IJ':9,
'IK':3,
'IM':4,
'JK':9,
'JN':8,
'KL':2,
'KM':4,
'KN':9,
'LM':2,
'LN':8,
'LO':9,
'MO':4,
'NO':7,
'NP':9,
'OP':5
}
vertex_list=[]
for f in edges_dict:
  vertex_list += f[0]
  vertex_list += f[1]
n=len(set(vertex_list))  
#print(edges_dict)
#G = nx.Graph()
#G.add_weighted_edges_from(edges)
#edge_labels = nx.get_edge_attributes(G, "weight")
#nx.draw_networkx_edge_labels(G,edge_labels)
#plt.show()

start_time=datetime.datetime.now()

treelists=[[{} for k in range(int(math.log2(n))+1)] for j in range(n)]

for i in range(n):
   #treelists[i][0]=[{} for j in range(n)]
    treelists[i][0]['']=0
    # print(treelists[i][0])     
for j in range(int(math.log2(n))):
  print("j=",j)
  for i in range(n):
  #treelists[i][0]=[{} for j in range(n)]
    for current_tree in treelists[i][j-1]:
      #print(treelists[i][j-1][current_tree])
      current_tree_cost=treelists[i][j-1][current_tree]
      #print(current_tree_cost)
      for e in edges_dict:
        if e[0] == vertex(i) or e[1] == vertex(i):
          other_vertex= e[1] if e[0] == vertex(i) else e[0]
          edge_cost=edges_dict[e]
          middle_edge = (vertex(i)+other_vertex)
          other_vertex_index=index(other_vertex)
          #print(other_vertex,other_vertex_index,edge_cost,middle_edge)
          for other_tree in treelists[other_vertex_index][j-1]:
            #ignore the tree if any vertex is part of current tree
            if set(current_tree).intersection(set(other_tree)):
              continue  
            #breakpoint()
            other_tree_cost=treelists[other_vertex_index][j-1][other_tree]
            newtree=str(current_tree)+middle_edge+other_tree
            newtree_cost=max(current_tree_cost,other_tree_cost)+edge_cost
            print(newtree,newtree_cost)
            treelists[i][j][newtree]=newtree_cost

j=int(math.log2(n))
trees=[[] for i in range(n)]
costs=[[] for i in range(n)]
sorted_cost_indices=[[] for i in range(n)]
sorted_trees=[[] for i in range(n)]
#convert to list and sort the trees 
for i in range(n):
  trees[i]=list(treelists[i][j-1].keys())
  costs[i]=list(treelists[i][j-1].values())
  sorted_cost_indices[i]=np.argsort(costs[i])
  sorted_trees[i]={trees[i][k]:costs[i][k] for k in sorted_cost_indices[i]}
  print(sorted_trees[i])
#raise SystemExit
print("j=",j)
global_minimum_cost=100000000
for i in range(n):
  costs_set=set({})
  for current_tree in sorted_trees[i]:
    current_tree_cost=sorted_trees[i][current_tree]
    if current_tree_cost > global_minimum_cost:
      continue
    for e in edges_dict:
      if e[0] == vertex(i) or e[1] == vertex(i):
        other_vertex= e[1] if e[0] == vertex(i) else e[0]
        edge_cost=edges_dict[e]
        middle_edge = (vertex(i)+other_vertex)
        other_vertex_index=index(other_vertex)
        lasttree_cost=1000000
        
        for other_tree in sorted_trees[other_vertex_index]:
          #ignore the tree if any vertex is part of current tree
          if set(current_tree).intersection(set(other_tree)):
            continue  

          other_tree_cost=sorted_trees[other_vertex_index][other_tree]
         
          if lasttree_cost < other_tree_cost:
            continue
          
          if other_tree_cost > current_tree_cost:
            if other_tree_cost+edge_cost in costs_set:
              break
            newtree_cost=other_tree_cost+edge_cost
          else:
            if current_tree_cost+edge_cost in costs_set:
              break
            newtree_cost=current_tree_cost+edge_cost  
          newtree=str(current_tree)+middle_edge+other_tree
          if newtree_cost < global_minimum_cost:
            global_minimum_cost=newtree_cost
          costs_set.add(newtree_cost)
          lasttree_cost=other_tree_cost
          print(newtree,newtree_cost)
          treelists[i][j][newtree]=newtree_cost
with open("output.txt",'a') as f:
  temp=0
  global_tree_list=[]
  for i in range(n):
    trees=list(treelists[i][int(math.log2(n))].keys())
    costs=list(treelists[i][int(math.log2(n))].values())
    sorted_cost_indices=np.argsort(costs)
    sorted_trees={trees[k]:costs[k] for k in sorted_cost_indices}
    if len(sorted_trees) > 0:
      global_tree_list+=[(trees[sorted_cost_indices[0]],costs[sorted_cost_indices[0]])]
    print(sorted_trees,file=f)
    temp+=len(sorted_trees)
  print("No. of trees:",temp,file=f)
  print(datetime.datetime.now()-start_time,"\n\n",file=f)
global_tree_list.sort(key=lambda x:x[1])
final_tree,final_root,final_cost=global_tree_list[0][0],global_tree_list[0][0][0],global_tree_list[0][1]
#print(final_tree,final_root,final_cost)
resultant_string=to_tree(final_tree,final_root)
print(resultant_string)
resultant_edges=resultant_string.split(',')
edges_final=[]
for i in resultant_edges:
  edges_final+=[(index(i[0]),index(i[1]))]
l=[]

f1 = open("layout-up", "w")
f2 = open("layout-down", "w")

for edge in edges_final:
#        print(edge[0],",",edge[1],sep="",end=" ")
  f1.write(str(edge[0])+","+str(edge[1])+"\n")
  l=[edge]+l
  
for edge in l:
  f2.write(str(edge[1])+","+str(edge[0])+"\n")