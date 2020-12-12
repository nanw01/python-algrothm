class Solution(object):
    def closest(self, array, target):
          # write your solution here
        if len(array) == 0:
            return -1

        left, right = 0, len(array)-1

        if target <= array[left]:
            return 0
        if target >= array[right]:
            return right

        while left < right:
            mid = left+(right-left)//2

            if target < array[mid]:


                if mid>0 and array[mid-1] >target:
                    if target - array[mid-1] > array[mid]-target:
                        return mid
                    else:
                        return mid-1
                right = mid
            elif target > array[mid]:
                
                if mid>0 and array[mid+1] >target:
                    if target - array[mid] > array[mid+1]-target:
                        return mid+1
                    else:
                        return mid
                left = mid+1

        return left
            
        

s = Solution()
a = [3, 4, 5, 6, 6, 12, 16]
b = 1
print(s.closest(a, b))
