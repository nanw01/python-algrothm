# 有重复的数

def subsets_recursive(nums):
    lst = []
    result=[]
    nums.sort()
    subsets_recursive_help(result,lst,nums,0)
    return result

def subsets_recursive_help(result,lst,nums,pos):
    result.append(lst[:])
    for i in range(pos,len(nums)):
        if(pos != i and nums[i]==nums[i-1]):
            continue
        lst.append(nums[i])
        subsets_recursive_help(result,lst,nums,i+1)
        lst.pop()


nums=['a','b','c']
print(subsets_recursive(nums))