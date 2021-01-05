
def _quick_sorted(nums: list) -> list:
    if len(nums) <= 1:
        return nums

    pivot = nums[0]
    left_nums = _quick_sorted([x for x in nums[1:] if x < pivot])
    right_nums = _quick_sorted([x for x in nums[1:] if x >= pivot])
    return left_nums + [pivot] + right_nums
