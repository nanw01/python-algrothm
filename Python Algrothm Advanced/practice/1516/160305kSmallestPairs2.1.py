# takes O(k) extra memory and O(k^2 log k) time
import heapq


def kSmallestPairs(nums1, nums2, k):
    n1 = nums1[:k+1]
    n2 = nums2[:k+1]
    return heapq.nsmallest(k, ([u, v] for u in n1 for v in n2), key=sum)


nums1 = [1, 7, 11]
nums2 = [2, 4, 6]
k = 4
print(kSmallestPairs(nums1, nums2, k))


nums1 = [1, 1, 2]
nums2 = [1, 2, 3]
k = 4
print(kSmallestPairs(nums1, nums2, k))
