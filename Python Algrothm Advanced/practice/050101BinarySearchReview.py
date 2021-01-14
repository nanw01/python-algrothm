# Binary Search Review
# 1.0.1  Ex.1: Binary Search Review
# Find 1st position of target, return -1 if not found

# How about last position, any position?

# Binary Search (iterative)


def bi_search_iter(alist,item):
    
    left ,right = 0,len(alist)-1

    while left<=right:
        mid = (left+right)//2
        if alist[mid]<item:
            left = mid+1
        elif alist[mid]>item:
            right = mid-1
        else:
            return mid
    return -1


if __name__ == '__main__':
    num_list = [1]
    print(bi_search_iter(num_list, 1))
    print(bi_search_iter(num_list, 4))

