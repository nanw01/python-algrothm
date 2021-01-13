class Solution(object):
    def deDup(self, input):
        """
        input: string input
        return: string
        """
        # write your solution here

        if len(input) == 0 or len(input) == 1:
            return input

        res = input[0]
        for i in range(1, len(input)):

            if input[i] == res[-1]:
                continue
            res += input[i]
        return res


print(Solution().deDup('aaaabbbc'))
