# 最长回文

class Solution():
    def longestPalindrome(self, s):
        if not s:
            return ''

        answer = (0, 0)
        for mid in range(len(s)):
            # 奇数
            answer = max(answer, self.get_palindrom_from(s, mid, mid))
            # 偶数
            answer = max(answer, self.get_palindrom_from(s, mid, mid+1))

        return s[answer[1]:answer[0]+answer[1]]

    def get_palindrom_from(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return (right-left-1, left + 1)


print(Solution().longestPalindrome("abcdzdcab"))
