from AdjListGraph import Graph
from AdjListGraph import Vertex

'''
递归方式进行图的深度优先查询
'''


def dfs(G, currentVert, visited):
    visited[currentVert] = True
    print(currentVert.getVertexID())
    for curr in currentVert.getConnections():
        if curr not in visited:
            dfs(G, curr, visited)


def DFSTraversal(G):
    visited = {}
    for current in G:
        if current not in visited:
            dfs(G, current, visited)


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
print(G.getEdges())
for k in G.getEdges():
    print(k)


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
print(G.getEdges())
for k in G.getEdges():
    print(k)


visited = {}
v = G.getVertex('e')
dfs(G, v, visited)
