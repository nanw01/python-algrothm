# ### Ex.1 Next Greater Element

# Given an array, print the Next Greater Element for every element. The Next greater Element for an element x is the first greater element on the right side of x in array. Elements for which no greater element exist, consider next greater element as -1.

# O(n^2)
def nextGreat(nums):
    if len(nums) == 0:
        return
    stack = []

    for i in range(len(nums)):
        stack.append('-1')
        for j in range(i+1,len(nums)):
            if nums[j]>nums[i]:
                stack.pop()
                stack.append(nums[j])
                break


    for i in range(len(nums)):
        print(nums[i],":",stack[i])
    



array = [6, 4, 5, 2, 25]
nextGreat(array)