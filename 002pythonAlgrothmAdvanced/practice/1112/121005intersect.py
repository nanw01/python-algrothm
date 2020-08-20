# ### Ex.5 Intersection of Two Arrays II
# Given two arrays, write a function to compute their intersection.
# Example:
# Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].
# Note:
# Each element in the result should appear as many times as it shows in both arrays.
# The result can be in any order.
# Follow up:
# What if the given array is already sorted? How would you optimize your algorithm?
# What if nums1's size is small compared to nums2's size? Which algorithm is better?
# What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?


def intersect(nums1, nums2):

    dict1 = dict()
    for i in nums1:
        if i not in dict1:
            dict1[i] = 1
        else:
            dict1[i] += 1
    ret = []
    for i in nums2:
        if i in dict1 and dict1[i] > 0:
            ret.append(i)
            dict1[i] -= 1
    return ret


nums1 = [1, 2, 2, 1]
nums2 = [2, 2]
print(intersect(nums1, nums2))
