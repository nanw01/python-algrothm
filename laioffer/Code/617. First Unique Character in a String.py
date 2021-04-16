# class Solution(object):
#     def firstUniqChar(self, input):
#         """
#         input: string input
#         return: int
#         """
#         # write your solution here

#         from collections import Counter

#         # counts = Counter(input)
#         # for i in input:
#         #     if counts.get(i) == 1:
#         #         return input.index(i)
#         # return -1

#         counts = Counter(input)
#         for i, j in enumerate(input):
#             if counts[j] == 1:
#                 return i
#         return -1
class Solution(object):
    def firstUniqChar(self, input):
        """
        input: string input
        return: int
        """
        # write your solution here
        hash = dict()
        for i in input:
            if i not in hash:
                hash[i] = 0
            else:
                hash[i] += 1
        print(hash)
        for i in range(len(input)):
            if hash[input[i]] == 0:
                return i

        return -1


s = Solution()
ss = "viv"
print(s.firstUniqChar(ss))
