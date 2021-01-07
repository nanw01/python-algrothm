# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def exist(self, root, target):
        """
        input: TreeNode root, int target
        return: boolean
        """
        # write your solution here
        return self._exist(root, 0, target)

    def _exist(self, node, sum, target):

        if not node:
            return False

        sum += node.val
        if not node.left and not node.right:
            return sum == target
        return self._exist(node.left, sum, target) or self._exist(node.right, sum, target)


t1 = TreeNode(0)
t2 = TreeNode(1)
t3 = TreeNode(2)

t1.left = t2
t1.right = t3


t4 = TreeNode(3)
t5 = TreeNode(4)

t2.left = t4
t2.right = t5

t6 = TreeNode(5)
t7 = TreeNode(0)

t3.left = t6
t3.right = t7


s = Solution()
print(s.exist(t1, 10))
