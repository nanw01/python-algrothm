class Solution(object):
    def subSets(self, set):
        """
        input : String set
        return : String[]
        """
        # write your solution here
        result = [[]]
        for num in set:
            for item in result[:]:
                x = item[:]
                x.append(num)
                result.append(x)
        return result


s = Solution()
print(s.subSets([1, 2, 3, 4]))
