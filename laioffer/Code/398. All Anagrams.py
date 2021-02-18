class Solution(object):
    def allAnagrams(self, sh, lo):
        """
        input: string sh, string lo
        return: Integer[]
        """
        # write your solution here
        res = []
        if sh is None or lo is None or len(sh) > len(lo):
            return res

        import copy

        counts = self._count(sh)
        temp = copy.deepcopy(counts)
        match = 0

        for i in range(len(lo)-len(sh)):
            for j in range(len(sh)):
                if lo[i+j] in temp:
                    temp[lo[i+j]] -= 1

                    if temp[lo[i+j]] == 0:
                        match += 1

                    if match == len(temp):
                        res.append(i + j - len(sh)+1)
                else:
                    break
            temp = copy.deepcopy(counts)
            match = 0

        return res

    def _count(self, sh):
        dic = {}
        for i in sh:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1

        return dic


print(Solution().allAnagrams("aab", "ababacbbaac"))
