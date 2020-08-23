class Vertex:
    def __init__(self, node):
        self.id = node
        # make all nodes unvisited
        self.visited = False

    def addNeighbor(self, neighbor, G):
        G.addEdge(self.id, neighbor)

    def getConnections(self, G):
        return G.adjMatrix[self.id]

    def getVertexID(self):
        return self.id

    def setVertexID(self, id):
        self.id = id

    def __str__(self):
        return str(self.id)


class Graph:
    def __init__(self, numVertices=10, directed=False):
        self.adjMatrix = [[None] * numVertices for _ in range(numVertices)]
        self.numVertices = numVertices
        self.vertices = []
        self.directed = directed
        for i in range(0, numVertices):
            newVertex = Vertex(i)
            self.vertices.append(newVertex)

    def addVertex(self, vtx, id):
        if 0 <= vtx < self.numVertices:
            self.vertices[vtx].setVertexID(id)

    def getVertex(self, n):
        for vertexin in range(0, self.numVertices):
            if n == self.vertices[vertexin].getVertexID():
                return vertexin
        return None

    def addEdge(self, frm, to, cost=0):
        if self.getVertex(frm) is not None and self.getVertex(to) is not None:
            self.adjMatrix[self.getVertex(frm)][self.getVertex(to)] = cost
            if not self.directed:
                self.adjMatrix[self.getVertex(to)][self.getVertex(frm)] = cost

    def getVertices(self):
        vertices = []
        for vertxin in range(0, self.numVertices):
            vertices.append(self.vertices[vertxin].getVertexID())
        return vertices

    def printMatrix(self):
        for u in range(0, self.numVertices):
            row = []
            for v in range(0, self.numVertices):
                row.append(
                    str(self.adjMatrix[u][v]) if self.adjMatrix[u][v] is not None else '/')
            print(row)

    def getEdges(self):
        edges = []
        for v in range(0, self.numVertices):
            for u in range(0, self.numVertices):
                if self.adjMatrix[u][v] is not None:
                    vid = self.vertices[v].getVertexID()
                    wid = self.vertices[u].getVertexID()
                    edges.append((vid, wid, self.adjMatrix[u][v]))
        return edges

    def getNeighbors(self, n):
        neighbors = []

        for v in range(0, self.numVertices):
            if n == self.vertices[v].getVertexID():
                for neighbor in range(0, self.numVertices):
                    if self.adjMatrix[v][neighbor] is not None:
                        neighbors.append(self.vertices[neighbor].getVertexID())

        return neighbors

    def isConnected(self, u, v):
        u_idx = self.getVertex(u)
        v_idx = self.getVertex(v)
        return self.adjMatrix[u_idx][v_idx] is not None

    def get2Hops(self, u):
        neighbors = self.getNeighbors(u)
        print(neighbors)
        hopset = set()
        for v in neighbors:
            hops = self.getNeighbors(v)
            hopset |= set(hops)
        return list(hopset)


if __name__ == "__main__":
    # add vertexs
    graph = Graph(6, True)
    graph.addVertex(0, 'a')
    graph.addVertex(1, 'b')
    graph.addVertex(2, 'c')
    graph.addVertex(3, 'd')
    graph.addVertex(4, 'e')
    graph.addVertex(5, 'f')
    graph.addVertex(6, 'g')  # doing nothing here
    graph.addVertex(7, 'h')  # doing nothing here

    print(graph.getVertices())

    # adde edges
    graph.addEdge('a', 'b', 1)
    graph.addEdge('a', 'c', 2)
    graph.addEdge('b', 'd', 3)
    graph.addEdge('b', 'e', 4)
    graph.addEdge('c', 'd', 5)
    graph.addEdge('c', 'e', 6)
    graph.addEdge('d', 'e', 7)
    graph.addEdge('e', 'a', 8)
    graph.addEdge('a', 'a', 18)
    print(graph.printMatrix())

    print(graph.getEdges())

    print('邻居：', graph.getNeighbors('a'))

    print(graph.isConnected('a', 'e'))

    print('get2Hops', graph.get2Hops('a'))

    G = Graph(5)
    G.addVertex(0, 'a')
    G.addVertex(1, 'b')
    G.addVertex(2, 'c')
    G.addVertex(3, 'd')
    G.addVertex(4, 'e')
    G.addEdge('a', 'e', 10)
    G.addEdge('a', 'c', 20)
    G.addEdge('c', 'b', 30)
    G.addEdge('b', 'e', 40)
    G.addEdge('e', 'd', 50)
    G.addEdge('f', 'e', 60)
    print(G.printMatrix())
    print(G.getEdges())
