def merge_sort(seq):
    if len(seq) <= 1:
        return seq

    mid = int(len(seq)/2)
    left = merge_sort(seq[:mid])
    right = merge_sort(seq[mid:])

    return merge_sorted_list(left, right)


def merge_sorted_list(sorted_a, sorted_b):

    len_a, len_b = len(sorted_a), len(sorted_b)
    a = b = 0
    new_sorted_seq = list()

    while a < len_a and b < len_b:
        if sorted_a[a] < sorted_b[b]:
            new_sorted_seq.append(sorted_a[a])
            a += 1
        else:
            new_sorted_seq.append(sorted_b[b])

    if a < len_a:
        new_sorted_seq.extend(sorted_a[a:])
    else:
        new_sorted_seq.extend(sorted_b[b:])

    return new_sorted_seq


l = [1, 3, 5, 7, 2, 2, 2, 2, 2, 2, 4, 6, 9, 8, 0, 11, -5]
l = merge_sort(l)
print(l)
