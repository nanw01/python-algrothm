import heapq


class Solution(object):
    def merge(self, arrayOfArrays):

        heap = []
        for i in range(len(arrayOfArrays)):
            if len(arrayOfArrays[i]):
                heap.append((arrayOfArrays[i][0], i, 0))
        print('----', heap)
        heapq.heapify(heap)

        result = []
        while heap:
            val, index_array, index_element = heapq.heappop(heap)
            print(val, index_array, index_element)
            result.append(val)
            if index_element + 1 < len(arrayOfArrays[index_array]):
                heapq.heappush(
                    heap,
                    (arrayOfArrays[index_array][index_element + 1],
                     index_array, index_element + 1)
                )
        return result


s = Solution()

lst = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10],
       [-9, - 7, - 4, 2, 4, 7, 5]]
ll = s.merge(lst)
print(ll)
