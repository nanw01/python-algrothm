# 插入排序
def insert_sort(items):
    for sort_inx in range(1, len(items)):
        unsort_inx = sort_inx
        while unsort_inx > 0 and items[unsort_inx-1] > items[unsort_inx]:
            items[unsort_inx-1], items[unsort_inx] = items[unsort_inx], items[unsort_inx-1]
            unsort_inx = unsort_inx-1
    return items


if __name__ == "__main__":
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    print(insert_sort(l))
