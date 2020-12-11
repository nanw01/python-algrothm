from AdjListGraph import Graph
from AdjListGraph import Vertex


import heapq

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.getVertexID())
        shortest(v.previous, path)
    return

def dijkstra(G, source, destination):
    print('''Dijkstra's shortest path''')
    # Set the distance for the source node to zero 
    source.setDistance(0)

    # Put tuple pair into the priority queue
    unvisitedQueue = [(v.getDistance(), v) for v in G]
    heapq.heapify(unvisitedQueue)

    while len(unvisitedQueue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisitedQueue)
        current = uv[1]
        current.setVisited()

        # for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            newDist = current.getDistance() + current.getWeight(next)
            
            if newDist < next.getDistance():
                next.setDistance(newDist)
                next.setPrevious(current)
                print('Updated : current = %s next = %s newDist = %s' \
                        % (current.getVertexID(), next.getVertexID(), next.getDistance()))
            else:
                print('Not updated : current = %s next = %s newDist = %s' \
                        % (current.getVertexID(), next.getVertexID(), next.getDistance()))

        # Rebuild heap
        # 1. Pop every item
        while len(unvisitedQueue):
            heapq.heappop(unvisitedQueue)
        # 2. Put all vertices not visited into the queue
        unvisitedQueue = [(v.getDistance(), v) for v in G if not v.visited]
        heapq.heapify(unvisitedQueue)


G = Graph(True)
G.addVertex('a')
G.addVertex('b')
G.addVertex('c')
G.addVertex('d')
G.addVertex('e')
G.addEdge('a', 'b', 4)  
G.addEdge('a', 'c', 1)
G.addEdge('c', 'b', 2)
G.addEdge('b', 'e', 4)
G.addEdge('c', 'd', 4)
G.addEdge('d', 'e', 4)

for v in G:
    for w in v.getConnections():
        vid = v.getVertexID()
        wid = w.getVertexID()
        print('( %s , %s, %3d)' % (vid, wid, v.getWeight(w)))


source = G.getVertex('a')
destination = G.getVertex('e')    
dijkstra(G, source, destination) 


for v in G.vertDictionary.values():
    print(source.getVertexID(), " to ", v.getVertexID(), "-->", v.getDistance())

path = [destination.getVertexID()]
shortest(destination, path)
print ('The shortest path from a to e is: %s' % (path[::-1]))