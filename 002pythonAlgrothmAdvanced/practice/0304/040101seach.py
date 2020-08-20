def find(nums,num):
    for i in nums:
        if i == num:
            return 0
    return -1

nums=[1,2,3,4,5,6,7,8,9]

print(find(nums,1))
print(find(nums,10))