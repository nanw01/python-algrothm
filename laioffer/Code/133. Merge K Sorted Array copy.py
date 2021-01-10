import heapq


class Solution(object):
    def merge(self, arrayOfArrays):
        """
        input: int[][] arrayOfArrays
        return: int[]
        """
        # write your solution here
        import heapq

        heap = []
        for i in range(len(arrayOfArrays)):
            if len(arrayOfArrays[i]):
                heap.append((arrayOfArrays[i][0], i, 0))
        heapq.heapify(heap)

        result = []
        while heap:
            val, index_array, index_element = heapq.heappop(heap)

            result.append(val)
            if index_element + 1 < len(arrayOfArrays[index_array]):
                heapq.heappush(
                    heap, (arrayOfArrays[index_array][index_element + 1], index_array, index_element + 1))
        return result


s = Solution()

lst = [[3, 2, 1, 5, 6, 4], [0, -2, -9, -4, -7, -8],
       [-43, -65, 897, 345, 3446, 234, -987, -325]]
ll = s.merge(lst)
print(ll)
heapq.heapify(ll)
while ll:
    print(heapq.heappop(ll), end=' ')
