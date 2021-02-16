class Solution(object):

    def validPalindrome(self, input):
        """
        input: string input
        return: boolean
        """
        # write your solution here
        return self._validPalindrome(list(input), False)

    def _validPalindrome(self, arr, del_mark):
        if not arr:
            return arr

        left, right = 0, len(arr)-1

        while left < right:

            if arr[left] != arr[right]:
                if del_mark:
                    return False
                else:
                    return self._validPalindrome(arr[left+1: right+1], True) or self._validPalindrome(arr[left:right], True)
            else:
                left += 1
                right -= 1

        return True


print(Solution().validPalindrome("oklvojceguiuooqfsvlappalvsfqoouiuigecjovlko"))
