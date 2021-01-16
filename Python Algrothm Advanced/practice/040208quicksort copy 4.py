# 来offer版本


from random import randrange


def partition(lst, start, end, pivot_index):
    lst[pivot_index], lst[end] = lst[end], lst[pivot_index]
    store_index = start
    pivot_value = lst[end]

    for i in range(start, end):
        if lst[i] < pivot_value:
            lst[i], lst[store_index] = lst[store_index], lst[i]
            store_index += 1
    lst[store_index], lst[end] = lst[end], lst[store_index]
    return store_index


def quickSortHelper(lst, start, end):
    if start >= end:
        return

    pivot_index = randrange(start, end+1)
    new_pivot = partition(lst, start, pivot_index, pivot_index)

    quickSortHelper(lst, start, new_pivot-1)
    quickSortHelper(lst, new_pivot+1, end)

    return lst


def quickSort(lst):
    return quickSortHelper(lst, 0, len(lst)-1)


alist = [28, 1, -1, 5]
quickSort(alist)
print(alist)
