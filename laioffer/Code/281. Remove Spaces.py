class Solution(object):
    def removeSpaces(self, input):
        """
        input: string input
        return: string
        """
        # write your solution here
        ss = input.split(' ')
        return ' '.join([i for i in ss if i != ''])


s = Solution()
print(s.removeSpaces('abb'))
