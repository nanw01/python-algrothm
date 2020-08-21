# def subsets(nums):
#     result=[[]]
#     for num in nums:
#         for element in result[:]:
#             x=element[:]
#             x.append(num)
#             result.append(x)
#     return result
def subsets_recursive(nums):

    lst=[]
    result=[]
    subsets_recursive_help(result,lst,nums,0)
    return result
    

def subsets_recursive_help(result,lst,nums,pos):
    result.append(lst[:])
    for i in range(pos,len(nums)):
        lst.append(nums[i])
        subsets_recursive_help(result,lst,nums,i+1)
        lst.pop()


nums=['a','b','c']
print(subsets_recursive(nums))






nums=['a','b','c']
# print(subsets(nums))

print(subsets_recursive(nums))
