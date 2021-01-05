from collections import Counter


class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        dic1 = Counter(nums1)
        dic2 = Counter(nums2)

        res = []
        for key, val in dic1.items():
            res += [key] * min(val, dic2[key])
        return res


s = Solution()
nums1 = [1, 2, 2, 1]
nums2 = [2, 2]

print(s.intersect(nums1, nums2))

dic1 = Counter(nums1)
dic2 = Counter(nums2)
