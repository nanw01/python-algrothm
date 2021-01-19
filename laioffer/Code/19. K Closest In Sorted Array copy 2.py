class Solution(object):
    def kClosest(self, array, target, k):
        """
        input: int[] array, int target, int k
        return: int[]
        """
        # write your solution here

        left, right = 0, len(array)-1
        while left+1 < right:
            mid = (left+right)//2

            if array[mid] < target:
                left = mid
            elif array[mid] >= target:
                right = mid

        closest = left if abs(
            array[left]-target) < abs(array[right]-target) else right

        res = []

        if k == 0:
            return res

        res.append(array[closest])
        left = closest-1
        right = closest+1
        total = len(array)
        while len(res) < k and (left >= 0 or right < total):
            if right < total and (left < 0 or abs(array[left]-target) > abs(array[right]-target)):
                res.append(array[right])
                right += 1
            elif left >= 0:
                res.append(array[left])
                left -= 1
        return res


if __name__ == "__main__":
    s = Solution()

    arr = [1, 3, 3, 6, 9, 9, 12, 15]
    target = 10
    k = 5

    print(s.kClosest(arr, target, k))
