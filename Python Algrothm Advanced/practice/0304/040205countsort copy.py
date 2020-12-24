def count_sort(items):
    #    确定范围
    mmax, mmin = items[0], items[0]
    for i in range(len(items)):
        mmax = max(items[i], mmax)
        mmin = min(items[i], mmin)

    llen = mmax - mmin + 1
    counts = [0]*llen

    # 计数
    for i in range(len(items)):
        counts[items[i]-mmin] = counts[items[i]-mmin] + 1

    # rr = []
    # # 打印
    # for i in range(llen):
    #     for _ in range(counts[i]):
    #         rr.append(i+mmin)
    # return rr
    # 排序
    pos = 0
    for i in range(llen):
        for _ in range(counts[i]):
            items[pos] = i+mmin
            pos += 1

    return items


l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0, 15]
print(count_sort(l))
