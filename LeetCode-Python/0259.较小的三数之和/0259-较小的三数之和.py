class Solution(object):
    def threeSumSmaller(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        res = 0
        nums.sort()
        for i, num in enumerate(nums):
            # ����Ҫ�ڱ�i����±��������ϳ�target - num �������������
            t = target - num
            left, right = i + 1, len(nums) - 1
            while left < right:
                if nums[left] + nums[right] >= t:
                    right -= 1 #��̫���ˣ����԰Ѻ���С
                elif nums[left] + nums[right] < t:
                    res += right - left #����������£�left�����롾left + 1�� right��������һ�������һ���
                    left += 1

        return res