# ### Ex.1 Next Greater Element

# Given an array, print the Next Greater Element for every element. The Next greater Element for an element x is the first greater element on the right side of x in array. Elements for which no greater element exist, consider next greater element as -1.


def nextGreat(nums):
    if len(nums) == 0:
        return
    stack = []
    stack.append(nums[0])
    
    for i in range(1, len(nums)):
        while (len(stack) != 0 and nums[i] > stack[-1]):
            num = stack.pop()
            print(num, ": ", array[i])
        stack.append(nums[i])
        
    while len(stack) != 0:
        print(stack.pop(), ": -1")



array = [6, 4, 5, 2, 25]
nextGreat(array)