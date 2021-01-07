class Solution(object):
    def rightShift(self, input, n):
        """
        input: string input, int n
        return: string
        """
        # write your solution here
        return ''.join([input[(i - n) % len(input)] for i in range(len(input))])


s = Solution()
print(s.rightShift('abc', 4))
