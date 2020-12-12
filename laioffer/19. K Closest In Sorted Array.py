class Solution:
    def kClosest(self, arr, x, k):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """
        # approach: use binary search to find the start which is closest to x
        left = 0
        right = len(arr) - k

        while left < right:
            mid = left + (right - left) // 2

            # mid + k is closer to x, discard mid by assigning left = mid + 1
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1

            # mid is equal or closer to x than mid + k, remains mid as candidate
            else:
                right = mid

        # left == right, which makes both left and left + k have same diff with x
        arr = arr[left : left + k]  
        # 排序
        for i in range(len(arr)):
            arr[i] = abs(arr[i]-x)
        
        for i in range(len(arr)):
            min = i
            for j in range(i+1,len(arr)):
                if arr[j] < min:
                    min = j
             
        






s = Solution()
array = [1, 4, 5]
target = 4
k=2
print(s.kClosest(array, target,k))
