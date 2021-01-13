class Solution(object):
    def closest(self, array, target):
        """
        input: int[] array, int target
        return: Integer[]
        """
        # write your solution here
        if len(array) <= 2:
            return array
        res = []
        left, right = 0, len(array)-1
        while left < right:
            sum = array[left]+array[right]
            print(target-sum)
            if sum < target:
                left += 1
            elif sum > target:
                right -= 1
            else:
                pass
            res.append([array[left], array[right]])
        return res[-3]


s = Solution()
print(s.closest([1, 4, 7, 13], 14))
