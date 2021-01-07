class Solution(object):
    def plus(self, digits):
        """
        input: int[] digits
        return: int[]
        """
        # write your solution here
        for i in range(len(digits)-1, -1, -1):
            if i == len(digits) - 1:
                digits[i] += 1

            if digits[i] == 10 and i != 0:
                digits[i] = 0
                digits[i - 1] += 1
            if digits[i] == 10 and i == 0:
                digits[i] = 0
                digits.insert(0, 1)

        return digits


s = Solution()
print(s.plus([9, 9, 9]))
