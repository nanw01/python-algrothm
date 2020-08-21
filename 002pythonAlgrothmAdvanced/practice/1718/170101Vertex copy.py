class Vertex:
    def __init__(self,node):
        pass

    def addNeighbor(self,neighber,G):
        pass

    def getConnections(self,G):
        pass

    def getVertexID(self):
        pass

    def setVertexID(self,id):
        pass

    def __str(self):
        pass

class  Graph:
    def __init__(self,numVertices=10,directed=False):
        self.adjMatrix = [[None]*numVertices for _ in range(numVertices)]
        self.numVertices = numVertices


    def addVertex(self,vtx,id):
        pass