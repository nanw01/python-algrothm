class Solution(object):
    def longestPalindrome(self, input):
        """
        input: string input
        return: string
        """
        # write your solution here
        start = 0
        end = 0

        for i in range(0, len(input)):
            len1 = self._helper(input, i, i)
            len2 = self._helper(input, i, i + 1)
            length = max(len1, len2)

            if length > end - start + 1:
                start = i - (length-1) // 2
                end = i + length//2

        return input[start:end+1]

    def _helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return r - l - 1


print(Solution().longestPalindrome('dbdcbdbabbacd'))
