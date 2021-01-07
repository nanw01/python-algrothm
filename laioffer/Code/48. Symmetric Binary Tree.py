# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.isSame(root, root)

    def isSame(self, node1, node2):
        if not node1 and not node2:
            return True
        if not node1 and node2:
            return False
        if node1 and not node2:
            return False
        return node1.val == node2.val and \
            self.isSame(node1.left, node2.right) and \
            self.isSame(node1.right, node2.left)


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
print(s.isSymmetric(t1))
