def _merge(a: list, b: list) -> list:
    """Merge two sorted list"""
    c = []
    while len(a) > 0 and len(b) > 0:
        if a[0] < b[0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])

    if len(a) == 0:
        c += b
    else:
        c += a
    return c


def _merge_sorted(nums: list) -> list:
    # Won't sort in place
    if len(nums) <= 1:
        return nums

    m = len(nums) // 2
    a = _merge_sorted(nums[:m])
    b = _merge_sorted(nums[m:])
    return _merge(a, b)


# Wrapper
def merge_sorted(nums: list, reverse=False) -> list:
    import time
    start = time.time()
    """Merge Sort"""
    nums = _merge_sorted(nums)
    if reverse:
        nums = nums[::-1]

    t = time.time() - start
    return nums, len(nums), t


l = [1, 3, 5, 7, 2, 4, 6, 9, 8, 0]
l = merge_sorted(l, reverse=False)[0]
print(l)
