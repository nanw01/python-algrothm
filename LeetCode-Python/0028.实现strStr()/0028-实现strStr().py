class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if not needle:
            return 0
        # if needle in haystack:
        #     return haystack.index(needle)
        for i, char in enumerate(haystack):
            if char == needle[0]:
                if haystack[i:i + len(needle)] == needle:
                    return i

        return -1


print(Solution().strstr('abbaabbab', 'bbab'))


a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(a[1:200])
