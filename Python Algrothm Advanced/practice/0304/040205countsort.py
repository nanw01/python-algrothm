def count_sort(items):
    # 计算范围

    mmax, mmin = items[0], items[0]

    for i in range(1, len(items)):
        if items[i] > mmax:
            mmax = items[i]
        elif items[i] < mmin:
            mmin = items[i]
        else:
            pass

    # 计数
    nums = mmax - mmin + 1
    counts = [0]*nums

    for i in range(len(items)):
        counts[items[i]-mmin] = counts[items[i]-mmin] + 1

    # 排序
    pos = 0
    for i in range(nums):
        for _ in range(counts[i]):
            items[pos] = i+mmin
            pos += 1

    return items


l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0, 15]
print(count_sort(l))
