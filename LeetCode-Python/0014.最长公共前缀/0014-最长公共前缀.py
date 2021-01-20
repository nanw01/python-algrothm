class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        strs.sort()

        res = ""
        pair = zip(strs[0],  strs[-1])
        print(pair)
        for x, y in pair:
            # print(x, y)
            if x == y:
                res += x
            else:
                break
        return res


print(Solution().longestCommonPrefix(["flow123", "flow456", "flow789"]))
