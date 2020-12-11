def merge_sort(seq):
    if len(seq) <= 1:
        return seq

    mid = int(len(seq)/2)
    left = merge_sort(seq[:mid])
    right = merge_sort(seq[mid:])

    return merge_sorted_list(left, right)


def merge_sorted_list(sorted_a, sorted_b):
    """ 合并两个有序序列，返回一个新的有序序列 
    :param sorted_a:
    :param sorted_b:
    """
    length_a, length_b = len(sorted_a), len(sorted_b)
    a = b = 0
    new_sorted_seq = list()

    while a < length_a and b < length_b:
        if sorted_a[a] < sorted_b[b]:
            new_sorted_seq.append(sorted_a[a])
            a += 1
        else:
            new_sorted_seq.append(sorted_b[b])
            b += 1

    # 最后别忘记把多余的都放到有序数组里        
    if a < length_a:
        new_sorted_seq.extend(sorted_a[a:])
    else:
        new_sorted_seq.extend(sorted_b[b:])

    return new_sorted_seq


l = [1, 3, 5, 7, 2, 2, 2, 2, 2, 2, 4, 6, 9, 8, 0, 11, -5]
l = merge_sort(l)
print(l)
