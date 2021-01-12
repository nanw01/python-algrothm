class Solution(object):
    def subSets(self, set):
        """
        input : String set
        return : String[]
        """
        # write your solution here

        result = [[]]

        for i in set:
            for r in result[:]:
                x = r[:]
                x.append(i)
                result.append(x)
        return result


print(Solution().subSets('abc'))
