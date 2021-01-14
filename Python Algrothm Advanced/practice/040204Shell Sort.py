# Shell Sort

def shell_sort(nums):

    gap = len(nums)
    length = len(nums)

    while (gap > 0):
        for i in range(gap, length):
            for j in range(i, gap - 1, -gap):
                if (nums[j - gap] > nums[j]):
                    nums[j], nums[j - gap] = nums[j - gap], nums[j]

        if (gap == 2): 
            gap = 1
        else:
            gap = gap // 2

    return nums


l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
l = shell_sort(l)
print(l)