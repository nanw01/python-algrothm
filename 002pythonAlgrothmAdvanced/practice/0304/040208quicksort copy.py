# quick sort

def quick_sorted(nums: list) -> list:
    if len(nums) <= 1:
        return nums

    pivot = nums[0]
    left_nums = quick_sorted([x for x in nums[1:] if x < pivot])
    right_nums = quick_sorted([x for x in nums[1:] if x >= pivot])

    return left_nums+[pivot]+right_nums


l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
l = quick_sorted(l)
print(l)
