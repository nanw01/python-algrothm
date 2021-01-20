# # 651. Common Numbers Of Two Arrays II(Array version)

# class Solution(object):
#     def common(self, A, B):
#         """
#         input: int[] A, int[] B
#         return: Integer[]
#         """
#         # write your solution here
#         from collections import Counter

#         ca = Counter(A)
#         cb = Counter(B)
#         res = []
#         for key, val in ca.items():
#             print(key, val)
#             res += [key]*min(val, cb[key])
#         return sorted(res)

#     def common1(self, A, B):
#         count_A = {}
#         for num in A:
#             count_A[num] = count_A.get(num, 0)+1
#         count_B = {}
#         for num in B:
#             count_B[num] = count_B.get(num, 0)+1

#         common = []
#         for num in count_A:
#             common += [num] * min(count_A[num], count_B.get(num, 0))

#         return sorted(common)


# print(Solution().common([1, 2, 3, 2], [3, 4, 2, 2, 2]))
# print(Solution().common1([1, 2, 3, 2], [3, 4, 2, 2, 2]))


class Solution(object):
    def common(self, A, B):
        """
        input: int[] A, int[] B
        return: Integer[]
        """
        count_A = {}
        for num in A:
            count_A[num] = count_A.get(num, 0)+1
        count_B = {}
        for num in B:
            count_B[num] = count_B.get(num, 0)+1

        common = []
        for num in count_A:
            common += [num] * min(count_A[num], count_B.get(num, 0))

        return sorted(common)


print(Solution().common([1, 2, 3, 2], [3, 4, 2, 2, 2]))
