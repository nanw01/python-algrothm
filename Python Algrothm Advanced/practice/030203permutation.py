def perm(result, nums, s):

    if (len(nums) == 0):
        s.append(result)

    for i in range(len(nums)):
        perm(result+str(nums[i]), nums[0:i]+nums[i+1:], s)  #

    return s


nums = [1, 2, 3, 4]
print(perm('', nums, []))
