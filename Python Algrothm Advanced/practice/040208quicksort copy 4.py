
def quick_sorted(nums: list) -> list:
    if len(nums) <= 1:
        return nums

    pivot = nums[0]
    left = quick_sorted([i for i in nums[1:] if i < pivot])
    right = quick_sorted([i for i in nums[1:] if i > pivot])

    return left+[pivot]+right


l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
l = quick_sorted(l)
print(l)
