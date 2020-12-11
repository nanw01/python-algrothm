# quick sort

def _quick_sorted(nums: list) -> list:
    if len(nums) <= 1:
        return nums

    pivot = nums[0]
    left_nums = _quick_sorted([x for x in nums[1:] if x < pivot])
    right_nums = _quick_sorted([x for x in nums[1:] if x >= pivot])
    return left_nums + [pivot] + right_nums


def quick_sorted(nums: list, reverse=False) -> list:
    """Quick Sort"""
    import time
    start = time.time()
    nums = _quick_sorted(nums)
    if reverse:
        nums = nums[::-1]
    t = time.time() - start
    return nums, len(nums), t


l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
l = quick_sorted(l, reverse=False)
print(l)
