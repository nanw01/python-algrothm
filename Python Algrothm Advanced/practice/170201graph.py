import sys


class Vertex():
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # set distance to infinity for all nodes.
        self.distance = sys.maxsize
        # make all nodes unvisited
        self.visited = False
        # Predecessor
        self.previous = None

    def addNeighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def getConnections(self):
        return self.adjacent.keys()

    def getVertexID(self):
        return self.id

    def getWeight(self, neighbor):
        return self.adjacent[neighbor]

    def setDistance(self, dist):
        self.distance = dist

    def getDistance(self):
        return self.distance

    def setPrevious(self, pre):
        self.previous = pre

    def setVisited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def __lt__(self, other):
        return self.distance < other.distance and self.id < other.id


class Graph:
    def __init__(self, directed=False):
        self.vertDictionary = {}
        self.numVertices = 0
        self.directed = directed

    def __iter__(self):
        return iter(self.vertDictionary.values())

    def isDirected(self):
        return self.isDirected

    def vectexCount(self):
        return self.numVertices

    def addVertex(self, node):
        self.numVertices += 1
        newVertex = Vertex(node)
        self.vertDictionary[node] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertDictionary:
            return self.vertDictionary[n]
        else:
            return None

    def addEdge(self, frm, to, cost=0):
        if frm not in self.vertDictionary:
            self.addVertex(frm)
        if to not in self.vertDictionary:
            self.addVertex(to)

        self.vertDictionary[frm].addNeighbor(self.vertDictionary[to], cost)
        if not self.directed:
            self.vertDictionary[to].addNeighbor(self.vertDictionary[frm], cost)

    def getVertices(self, current):
        return self.vertDictionary.keys()

    def getPrevious(self, current):
        self.previous = current

    def getEdges(self):
        edges = []
        for _, currentVert in self.vertDictionary.items():
            for nbr in currentVert.getConnections():
                currentVertID = currentVert.getVertexID()
                nbrID = nbr.getVertexID()
                edges.append(
                    (currentVertID, nbrID, currentVert.getWeight(nbr)))
        return edges

    def getNeighbors(self, v):
        vertex = self.vertDictionary[v]
        return vertex.getConnections()


if __name__ == '__main__':

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
    print(G.getEdges())

    for k in G.getEdges():
        print(k)
