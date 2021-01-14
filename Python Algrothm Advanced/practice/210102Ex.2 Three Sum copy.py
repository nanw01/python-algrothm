class Solution(object):
    def allTriples(self, array, target):

        res = []
        array.sort()
        
        for i in range(len(array)-2):
            if i > 0 and array[i] == array[i-1]:
                continue
            l, r = i+1, len(array)-1
            while l < r:
                s = array[i] + array[l] + array[r]
                if s < target:
                    l += 1
                elif s > target:
                    r -= 1
                else:
                    res.append((array[i], array[l], array[r]))
                    while l < r and array[l] == array[l+1]:
                        l += 1
                    while l < r and array[r] == array[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res


print(Solution().allTriples([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3))
