def merge_sorted(nums):
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    a = merge_sorted(nums[:mid])
    b = merge_sorted(nums[mid:])
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


l = [1, 3, 5, 7, 2, 4, 6, 9, 8, 0]
# l = [12, 11, 13, 5, 6, 7]
l = merge_sorted(l)
print(l)
