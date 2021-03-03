import heapq


class Solution(object):
    def merge(self, arrayOfArrays):
        heap = []
        for i in range(len(arrayOfArrays)):
            if len(arrayOfArrays[i]):
                heap.append((arrayOfArrays[i][0], i, 0))
        print(heap)
        heapq.heapify(heap)
        ret = []
        while heap:
            val, i, j = heapq.heappop(heap)
            # print(val, i, j)
            ret.append(val)
            if j+1 < len(arrayOfArrays[i]):
                print((arrayOfArrays[i][j+1], i, j+1))
                heapq.heappush(heap, (arrayOfArrays[i][j+1], i, j+1))

        return ret


# s = Solution()

# lst = [[3, 2, 1, 5, 6, 4], [0, -2, -9, -4, -7, -8]]
# # print(s.merge(lst))
# class Solution(object):
print(Solution().merge([[1, 0], [-2, -9], [9, -9, 7]]))
