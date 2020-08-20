from heapq import heappush, heappop
# heap = []
# data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
# for item in data:
#     heappush(heap, item)



# ordered = []
# while heap:
#     ordered.append(heappop(heap))

# print('ordered',ordered)
# # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# data.sort()
# data == ordered

import heapq

data = [1,5,3,2,8,5]
heapq.heapify(data)
print(data)

# while data:
#     print(data)
#     print(heappop(data))
