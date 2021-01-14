def _merge_sorted(nums):
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = _merge_sorted(nums[:mid])
    right = _merge_sorted(nums[mid:])

    new = []

    while len(left) > 0 and len(right) > 0:
        if left[0] < right[0]:
            new.append(left[0])
            left = left[1:]
        else:
            new.append(right[0])
            right = right[1:]

    if len(left) == 0:
        new += right
    else:
        new += left
    return new


# l = [1, 3, 5, 7, 2, 4, 6, 9, 8, 0]
l = [12, 11, 13, 5, 6, 7]
l = _merge_sorted(l)
print(l)
