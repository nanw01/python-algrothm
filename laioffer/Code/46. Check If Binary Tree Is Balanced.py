# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        if not root.left and not root.right:
            return True

        def getHeight(node, h):
            if not node:
                return h
            return max(getHeight(node.left, h + 1), getHeight(node.right, h + 1))
        return self.isBalanced(root.left) and self.isBalanced(root.right) and abs(getHeight(root.left, 0) - getHeight(root.right, 0)) <= 1


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
print(s.isBalanced(t1))
