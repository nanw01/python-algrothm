import heapq


class Solution(object):
    def smallestRange(self, arrays):
        heap = []
        k = len(arrays)
        maxval = float('-inf')
        for i in range(len(arrays)):
            if len(arrays[i]):
                heap.append((arrays[i][0], i, 0))
                maxval = max(maxval, arrays[i][0])

        minrange = [float('-inf'), float('inf')]

        heapq.heapify(heap)
        while len(heap) == k:
            minval, i, j = heapq.heappop(heap)
            if maxval - minval < minrange[1] - minrange[0]:
                minrange = [minval, maxval]

            if j+1 < len(arrays[i]):
                maxval = max(maxval, arrays[i][j+1])
                heapq.heappush(heap, (arrays[i][j+1], i, j+1))

        return minrange


aa = [
    [22, 251, 640, 699, 837, 974],
    [149, 327],
    [604],
    [88, 103, 175, 225, 307, 385, 419, 427, 490,
        586, 647, 707, 767, 796, 817, 901, 909],
    [507, 603],
    [116, 190, 208, 306, 314, 386, 505, 643, 903, 985],
    [31, 243, 287, 341, 381, 461, 549, 608, 837, 897, 972],
    [214, 257, 350, 417, 435, 537, 618, 757, 831, 858, 875, 921, 963, 970, 982],
    [44, 71, 75, 198, 216, 235, 411, 493, 746, 800, 835, 878, 920],
    [71, 98, 161, 338, 347, 389, 394, 437, 525, 549, 729, 763, 821, 878, 987]
]
s = Solution()
print(s.smallestRange(aa))
