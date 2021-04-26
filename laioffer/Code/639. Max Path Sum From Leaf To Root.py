# # Definition for a binary tree node.
# # class TreeNode(object):
# #     def __init__(self, x):
# #         self.val = x
# #         self.left = None
# #         self.right = None
# class Solution(object):
#     def maxPathSumLeafToRoot(self, root):
#         """
#         input: TreeNode root
#         return: int
#         """
#         # write your solution here
#         self._max_sum = float('-inf')
#         self._maxPathSumLeafToRoot(root, 0)
#         return self._max_sum

#     def _maxPathSumLeafToRoot(self, node, cur):
#         if node is None:
#             return float('-inf')

#         if node.left is None and node.right is None:
#             self._max_sum = max(self._max_sum, cur + node.val)
#             return

#         self._maxPathSumLeafToRoot(node.left, cur + node.val)
#         self._maxPathSumLeafToRoot(node.right, cur + node.val)

#         return


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def maxPathSumLeafToRoot(self, root):
        """
        input: TreeNode root
        return: int
        """
        # write your solution here
        self.maxVal = float('-inf')
        self.tempVal = 0
        self._maxPathSumLeafToRoot(root)
        return self.maxVal

    def _maxPathSumLeafToRoot(self, node):
        if not node:
            return

        self.tempVal = self.tempVal + node.val
        print('a:', self.tempVal)
        if not node.left and not node.right:
            self.maxVal = max(self.maxVal, self.tempVal)

        self._maxPathSumLeafToRoot(node.left)
        self._maxPathSumLeafToRoot(node.right)
        self.tempVal -= node.val


a = TreeNode(10)
b = TreeNode(-2)
c = TreeNode(7)
d = TreeNode(8)
e = TreeNode(-4)

a.left = b
a.right = c
b.left = d
b.right = e

print(Solution().maxPathSumLeafToRoot(a))
