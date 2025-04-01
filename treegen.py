#import networkx as nx 
#import matplotlib.pyplot as plt
import argparse

def largest_power_le(n):
    k=0
    while(n>=pow(2,k)):
       k+=1
    return pow(2,k-1)
     
def generate_tree(n,i):
  if(n<=3):
    print("Error, n<=3")
    return (0,[])
  else:
    k=largest_power_le(n)
    if(k==n):
         if(n==4):
           return (i+3,[(i,i+1),(i+1,i+3),(i+2,i+3)])
         else:
           r1,l1=generate_tree(n//2,i)
           r2,l2=generate_tree(n//2,i+n//2)
           return (n+i-1,l1+l2+[(n//2+i-1,i+n-1)])
    else:
        if(n-k<4):
           r,l=generate_tree(k,i)
           if(n-k == 1):
             l.remove((k+i-2,k+i-1))
             l+=[(k+i-2,k+i)]
             l+=[(k+i,k+i-1)]
           elif(n-k == 2):
             l.remove((k+i-2,k+i-1))
             l+=[(k+i-2,k+i)]
             l+=[(k+i,k+i+1)]
             l+=[(k+i+1,k+i-1)]
           else:
             l.remove((k+i-3,k+i-1))
             l.remove((k+i-2,k+i-1))
#             l+=[(k+i-3,k+i)]
#             l+=[(k+i,k+i-1)]
             l+=[(k+i-3,k+i+2)]
             l+=[(k+i+2,k+i-1)]


             l+=[(k+i-2,k+i)]
             l+=[(k+i,k+i+1)]
             l+=[(k+i+1,k+i-1)]
             
#             l+=[(k+i-2,k+i+1)]
#             l+=[(k+i+1,k+i+2)]
#             l+=[(k+i+2,k+i-1)]
           return (r,l)
        else:
           r1,l1=generate_tree(k,i)   
           r2,l2=generate_tree(n-k,k+i)
           return r1,l1+l2+[(r2,r1)]  

          
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
     f1 = open("layout-up", "w")
     f2 = open("layout-down", "w")
     f3 = open("layout", "w")
#     print("n=",n)
     for edge in t:
#        print(edge[0],",",edge[1],sep="",end=" ")
        f1.write(str(edge[0])+","+str(edge[1])+"\n")
        l=[edge]+l
        t1=t1+[edge]+[(edge[1],edge[0])]
     for edge in l:
#        print(edge[1],",",edge[0])
        f2.write(str(edge[1])+","+str(edge[0])+"\n")
     for edge in sorted(t1):
#        print(edge[1],",",edge[0])
        f3.write(str(edge[0])+","+str(edge[1])+"\n")

     f1.close()
     f2.close()
     f3.close()
#     print("")

