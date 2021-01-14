class Solution(object):
    def topKFrequent(self, combo, k):
        """
        input: string[] combo, int k
        return: string[]
        """
        # write your solution here
        from collections import Counter
        c = Counter(combo)

        return [key for key, value in c.most_common(k)]


print(Solution().topKFrequent(
    ["d", "a", "c", "b", "d", "a", "b", "b", "a", "d", "d", "a", "d"], 5))
