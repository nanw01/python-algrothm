# ### Ex.3: Kth Smallest Element in a Sorted Matrix
# Given a n x n matrix where each of the rows and columns are sorted in ascending order,
# find the kth smallest element in the matrix.
# Note that it is the kth smallest element in the sorted order, not the kth distinct element.


from bisect import bisect


def kthSmallest(matrix, k):

    pass


matrix = [
    [1, 4, 8, 10, 15],
    [3, 5, 6, 7, 20],
    [9, 20, 22, 24, 29],
    [11, 22, 23, 29, 39]
]
k = 5
kthSmallest(matrix, k)
