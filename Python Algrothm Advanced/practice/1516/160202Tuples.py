from heapq import heappop,heappush
heap = []
data = [(1, 'J'), (4, 'N'), (3, 'H'), (2, 'O')]
for item in data:
    heappush(heap, item)

while heap:
    item = heappop(heap) 
    print(item[0], ": ", item[1])