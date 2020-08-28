# 冒泡排序

def _bubble_sort(nums: list, reverse=False):
    for i in range(len(nums)):
        for j in range(len(nums)-i-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]

    if reverse:
        nums.reverse()
    return len(nums)


def bubble_sorted(nums: list, reverse=False):
    nums_copy = list(nums)
    _bubble_sort(nums_copy, reverse=reverse)
    return nums_copy

    
if __name__ == "__main__":
    
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    l = bubble_sorted(l, False)
    print(l)
