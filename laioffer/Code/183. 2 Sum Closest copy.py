class Solution(object):
    def closest(self, array, target):
        """
        input: int[] array, int target
        return: Integer[]
        """
        # write your solution here
        if len(array) <= 2:
            return array
        array = sorted(array)
        left, right = 0, len(array)-1
        res = [[array[left], array[right]]]
        while left < right:

            res.append([array[left], array[right]])
            print(left, right, res)
            print(abs(target - sum(res[-2])), abs(target - sum(res[-1])))
            if abs(target - sum(res[-2])) < abs(target - sum(res[-1])):
                return res[-2]
            if array[left]+array[right] < target:
                left += 1
            elif array[left]+array[right] > target:
                right -= 1
            else:
                return [array[left], array[right]]
        if abs(target - sum(res[-2])) <= abs(target - sum(res[-1])):
            return res[-2]
        if abs(target - sum(res[-2])) > abs(target - sum(res[-1])):
            return res[-1]


s = Solution()
print(s.closest([2, -3, 9], 4))
