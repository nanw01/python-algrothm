from AdjListGraph import Graph
from AdjListGraph import Vertex


G = Graph(True)
G.addVertex('a')
G.addVertex('b')
G.addVertex('c')
G.addVertex('d')
G.addVertex('e')
G.addVertex('f')
G.addEdge('a', 'b', 1)  
G.addEdge('a', 'c', 2)
G.addEdge('b', 'd', 3)
G.addEdge('b', 'e', 4)
G.addEdge('c', 'd', 5)
G.addEdge('c', 'e', 6)
G.addEdge('d', 'e', 7)
G.addEdge('e', 'a', 8)
G.addEdge('a', 'f', 9)
print (G.getEdges())
for k in G.getEdges():
    print(k)



def dfs(G, currentVert, visited):
    visited[currentVert] = True  # mark the visited node 
    print("traversal: " + currentVert.getVertexID())
    for nbr in currentVert.getConnections():  # take a neighbouring node 
        if nbr not in visited:  # condition to check whether the neighbour node is already visited
            dfs(G, nbr, visited)  # recursively traverse the neighbouring node
    return 
 
def DFSTraversal(G):
    visited = {}  # Dictionary to mark the visited nodes 
    for currentVert in G:  # G contains vertex objects
        if currentVert not in visited:  # Start traversing from the root node only if its not visited 
            dfs(G, currentVert, visited)  # For a connected graph this is called only onc


DFSTraversal(G)


G = Graph(True)
G.addVertex('a')
G.addVertex('b')
G.addVertex('c')
G.addVertex('d')
G.addVertex('e')
G.addVertex('f')
G.addEdge('a', 'b', 1)
G.addEdge('a', 'c', 2)
G.addEdge('b', 'd', 3)
G.addEdge('b', 'e', 4)
G.addEdge('c', 'd', 5)
G.addEdge('c', 'e', 6)
G.addEdge('d', 'e', 7)
G.addEdge('e', 'a', 8)
G.addEdge('a', 'f', 9)
print (G.getEdges())
for k in G.getEdges():
    print(k)


visited = {}
v = G.getVertex('e')
dfs(G, v, visited)