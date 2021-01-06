class Solution(object):
    def reverseWords(self, input):
        """
        input: string input
        return: string
        """
        # write your solution here

        s = [i for i in input.split(' ') if i != '']
        s = s[::-1]

        return ' '.join(s)


s = Solution()
ss = " I  love  Google  "
print(s.reverseWords(ss))
