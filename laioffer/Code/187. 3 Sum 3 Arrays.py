class Solution(object):
    def exist(self, a, b, c, target):
        """
        input: int[] a, int[] b, int[] c, int target
        return: boolean
        """
        # write your solution here
        for i in a:
            for j in b:
                if (target - i - j) in set(c):
                    return True

        return False
