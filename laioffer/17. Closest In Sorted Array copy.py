class Solution(object):
    def closest(self, array, target):
        if len(array) ==0 or array is None:
            return -1
        elif array[0] >= target:
            return 0
        elif array[len(array)-1] <= target:
            return len(array)-1
        else:
            left, right = 0, len(array)-1
            while left < right:
                mid = left+(right-left)//2
                if array[mid] < target:

                    if mid < len(array)-1 and array[mid+1] > target:
                        if array[mid+1]-target > target - array[mid]:
                            return mid
                        else:
                            return mid+1
                    left = mid+1
                elif array[mid] > target:
                    if mid > 0 and array[mid-1] < target:
                        if array[mid]-target > target - array[mid-1]:
                            return mid-1
                        else:
                            return mid

                    right = mid-1
                else:
                    return mid
            return left


s = Solution()
a = [1, 4, 6]
b = 3
print(s.closest(a, b))
