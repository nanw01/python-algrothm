from AdjListGraph import Graph
from AdjListGraph import Vertex


def dfsIterative(G, start, dest):
    stack = []  # vertex
    visited = set()  # vertex id
    parent = {}  # vertex id
    stack.append(start)
    while len(stack) != 0:
        curr = stack.pop()  # vertex
        print("visiting ", curr.getVertexID())
        if (curr.getVertexID() == dest.getVertexID()):
            return parent
        neighbors = G.getNeighbors(curr.getVertexID())
        for n in neighbors:
            id = n.getVertexID()
            if id in visited:
                continue
            visited.add(id)
            parent[id] = curr.getVertexID()
            stack.append(n)
    return None


G = Graph(True)
G.addVertex('a')
G.addVertex('b')
G.addVertex('c')
G.addVertex('d')
G.addVertex('e')
G.addVertex('f')
G.addEdge('a', 'b', 1)
G.addEdge('a', 'c', 1)
G.addEdge('b', 'd', 1)
G.addEdge('b', 'e', 1)
G.addEdge('c', 'd', 1)
G.addEdge('c', 'e', 1)
G.addEdge('d', 'e', 1)
G.addEdge('e', 'a', 1)
G.addEdge('a', 'f', 1)


start = G.getVertex('a')
dest = G.getVertex('e')
parent = dfsIterative(G, start, dest)
print('parent', parent)


G = Graph(True)
G.addVertex('a')
G.addVertex('b')
G.addVertex('c')
G.addVertex('d')
G.addVertex('e')
G.addVertex('f')
G.addEdge('a', 'b', 1)
G.addEdge('a', 'c', 1)
G.addEdge('a', 'd', 1)
G.addEdge('d', 'e', 1)
G.addEdge('e', 'f', 1)
print(G.getEdges())
for k in G.getEdges():
    print(k)


start = G.getVertex('a')
dest = G.getVertex('c')
parent = dfsIterative(G, start, dest)
print(parent)
