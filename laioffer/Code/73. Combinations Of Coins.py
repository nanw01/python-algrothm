class Solution(object):
    def combinations(self, target, coins):
        """
        input: int target, int[] coins
        return: int[][]
        """
        # write your solution here
        res = []
        cur = []
        self._combination(0, target, coins, cur, res)
        return res

    def _combination(self, index, target, coins, cur, res):
        if index == len(coins) - 1
