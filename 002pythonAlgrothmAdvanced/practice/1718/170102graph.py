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
        return str(self.id)+' adjacent: ' + str([x.id for x in self.adjacent])

    def __lt__(self, other):
        return self.distance < other.distance and self.id < other.id


class Graph:
    def __init__(self, directed=False):
        pass

    def __iter__(self):
        pass

    def isDirected(self):
        pass

    def vectexCount(self):
        pass

    def addVertex(self, node):
        pass

    def getVertex(self, n):
        pass

    def addEdge(self, frm, to, cost=0):
        pass

    def getVertices(self, current):
        pass

    def getPrevious(self, current):
        pass

    def getEdge(self):
        pass

    def getNeighbors(self, v):
        pass
