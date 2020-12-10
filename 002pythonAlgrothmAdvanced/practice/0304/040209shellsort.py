def shell_sort(nums):
    gap = len(nums)
    length = len(nums)
    while gap > 0:
        print( gap,length)

        for i in range(gap,length):
            print(gap,i)
            for j in range(i,gap-1,-gap):
                print(gap,j,j-gap)
                if nums[j-gap]>nums[j]:
                    nums[j],nums[j-gap] = nums[j-gap],nums[j]
        if gap == 2:
            gap=1
        else:
            gap = gap//2
    return nums

if __name__ == "__main__":
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    print(shell_sort(l))