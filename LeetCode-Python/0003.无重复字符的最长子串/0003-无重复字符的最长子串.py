class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, right = 0, 0
        dic = dict()
        res = 0
        while right < len(s):
            if s[right] in dic:
                left = max(left, dic[s[right]] + 1)
            dic[s[right]] = right
            res = max(res, right - left + 1)
            right += 1
        return res


# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         if not s:
#             return 0
#         if len(s) == 1:
#             return 1

#         l = 0
#         r = 1
#         maxl = 1
#         pool = set()
#         pool.add(s[l])
#         while r < len(s):
#             if s[r] in pool:
#                 maxl = max(maxl, r-l)
#                 l += 1
#                 r = l+1
#                 pool.clear()
#                 pool.add(s[l])
#             else:
#                 pool.add(s[r])
#                 r += 1
#                 maxl = max(maxl, r-l)

#         return maxl
