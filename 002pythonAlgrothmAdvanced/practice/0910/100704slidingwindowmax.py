# ### Ex.4 Sliding Window Max
# Given an array and an integer k, find the maximum for each and every contiguous subarray of size k
# Examples:
# Input :
# arr[] = {1, 2, 3, 1, 4, 5, 2, 3, 6}
# k = 3
# Output :
# 3 3 4 5 5 5 6
# Input :
# arr[] = {8, 5, 10, 7, 9, 4, 15, 12, 90, 13}
# k = 4
# Output :
# 10 10 10 15 15 90 90

from collections import deque

def movingMax(arr,k):
    n = len(arr)
    Qi = deque()
    
    # Process first k (or first window) 
    # elements of array
    for i in range(k):
        # For every element, the previous 
        # smaller elements are useless
        # so remove them from Qi
        while Qi and arr[i] >= arr[Qi[-1]] :
            Qi.pop()
        
        # Add new element at rear of queue
        Qi.append(i)
        
    # Process rest of the elements, i.e. 
    # from arr[k] to arr[n-1]
    for i in range(k, n):
        
        # The element at the front of the
        # queue is the largest element of
        # previous window, so print it
        print(str(arr[Qi[0]]) + " ", end = "")
        
        # Remove the elements which are 
        # out of this window
        while Qi and Qi[0] <= i-k:
            
            # remove from front of deque
            Qi.popleft() 
        
        # Remove all elements smaller than
        # the currently being added element 
        # (Remove useless elements)
        while Qi and arr[i] >= arr[Qi[-1]] :
            Qi.pop()
        
        # Add current element at the rear of Qi
        Qi.append(i)
    
    # Print the maximum element of last window
    print(str(arr[Qi[0]]))



arr = [12, 1, 78, 90, 57, 89, 56]
k = 3
movingMax(arr, k)


