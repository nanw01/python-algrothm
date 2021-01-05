
def mergeSort(array):
    """
    input: int[] array
    return: int[]
    """
    # write your solution here

    if len(array) <= 1:
        return array

    mid = len(array)//2
    l = mergeSort(array[:mid])
    r = mergeSort(array[mid:])

    c = []

    while len(l) > 0 and len(r) > 0:
        if l[0] < r[0]:
            c.append(l[0])
            l.remove(l[0])
        else:
            c.append(r[0])
            r.remove(r[0])

    if len(l) == 0:
        c += r
    else:
        c += l

    return c


# l = [1, 3, 5, 7, 2, 4, 6, 9, 8, 0]
l = [12, 11, 13, 5, 6, 7]
l = mergeSort(l)
print(l)
