#import networkx as nx 
#import matplotlib.pyplot as plt
import math
import argparse

def smallest_power_ge(n):
    k=int(math.log2(n))
    if(n>pow(2,k)):
       k+=1
    return pow(2,k)
     
def generate_tree(n,i):
  if(n<2):
    print("Error, At least 2 nodes needed..")
    return (0,[])
  else:
    if (n==2):
        return (i+1,[(i,i+1)])
    else:
        k=smallest_power_ge(n)
        r1,l1=generate_tree(k//2,i)
        r2,l2=generate_tree(k//2,i+k//2)
        root=k+i-1
        edges=l1+l2+[(r1,r2)]
#        breakpoint()
        if (n!=k):
            e=k-n
            for index in range(e):
              node = 2*index
              parent = node + 1
              edges.remove((node,parent))
        return (root,edges)
   
          
if __name__ == "__main__":
#     n=input("Enter n")
     parser = argparse.ArgumentParser()
     parser.add_argument("--n", type=int)
     args = parser.parse_args()
     n = int(args.n)
     
     r,t=generate_tree(int(n),0)
#     print(r,t)
     l=[]
     t1=[]
#     f1 = open("layout-up", "w")
#     f2 = open("layout-down", "w")
#     f3 = open("layout", "w")
#     print("n=",n)
     for edge in t:
        print(edge[0],",",edge[1],sep="",end=" ")
#        f1.write(str(edge[0])+","+str(edge[1])+"\n")
#        l=[edge]+l
#        t1=t1+[edge]+[(edge[1],edge[0])]
#     for edge in l:
#        print(edge[1],",",edge[0])
#        f2.write(str(edge[1])+","+str(edge[0])+"\n")
#     for edge in sorted(t1):
#        print(edge[1],",",edge[0])
#        f3.write(str(edge[0])+","+str(edge[1])+"\n")

#     f1.close()
#     f2.close()
#     f3.close()
#     print("")

