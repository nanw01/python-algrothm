def permSizeK(result, nums, k):
    if k == 0:
        print(result)
    for i in range(len(nums)):
        permSizeK(result+str(nums[i]), nums[0:i] + nums[i+1:], k - 1)



nums = [1, 2, 3, 4]
k = 2
permSizeK('', nums, k)