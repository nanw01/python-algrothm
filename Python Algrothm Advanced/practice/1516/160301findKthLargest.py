import heapq

# O(k+(n-k)lgk) time, min-heap
def findKthLargest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heapq.heappop(heap)

nums = [5,11,3,6,12,9,8,10,14,1,4,2,7,15]
k = 5
print(findKthLargest(nums, k))


# O(k+(n-k)lgk) time, min-heap        
def findKthLargest1(nums, k):
    return heapq.nlargest(k, nums)[k-1]

nums = [5,11,3,6,12,9,8,10,14,1,4,2,7,15]
k = 5
print(findKthLargest1(nums, k))