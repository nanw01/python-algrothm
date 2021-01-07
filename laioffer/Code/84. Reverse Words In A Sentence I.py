class Solution(object):
    def reverseWords(self, input):
        """
        input: string input
        return: string
        """
        # write your solution here
        return ' '.join(input.split(' ')[::-1])


s = Solution()
print(s.reverseWords('I love Google'))
