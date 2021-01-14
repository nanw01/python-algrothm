# ### <a id='Ex4'>Ex4：Shuffle Array</a>
# Given an array of 2n elements in the following format { a1, a2, a3, a4, ….., an, b1, b2, b3, b4, …., bn }. 
# The task is shuffle the array to {a1, b1, a2, b2, a3, b3, ……, an, bn } without using extra space.
# ** Examples: **
# Input : arr[] = { 1, 2, 9, 15 }
# Output : 1 9 2 15
# Input :  arr[] = { 1, 2, 3, 4, 5, 6 }
# Output : 1 4 2 5 3 6


#  从中间分两半，从中间交换，在分再换

def shufleArray(a, left, right):
 
    # If only 2 element, return
    if (right - left == 1):
        return
 
    # Finding mid to divide the array
    mid = (left + right) // 2
 
    # Using temp for swapping first
    # half of second array
    temp = mid + 1
 
    # Mid is use for swapping second
    # half for first array
    mmid = (left + mid) // 2
 
    # Swapping the element
    for i in range(mmid + 1, mid + 1):
        (a[i], a[temp]) = (a[temp], a[i])
        temp += 1
 
    # Recursively doing for 
    # first half and second half
    shufleArray(a, left, mid)
    shufleArray(a, mid + 1, right)


a = [1, 3, 5, 7, 2, 4, 6, 8] 
n = len(a) 
shufleArray(a, 0, n - 1)
print(a)
