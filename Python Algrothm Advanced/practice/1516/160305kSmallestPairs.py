import itertools
def kSmallestPairs(nums1, nums2, k):
    return sorted(itertools.product(nums1, nums2), key=sum)[:k]


nums1 = [1,7,11]
nums2 = [2,4,6]
k = 4
print(kSmallestPairs(nums1, nums2, k))



nums1 = [1,1,2]
nums2 = [1,2,3]
k = 4
print(kSmallestPairs(nums1, nums2, k))