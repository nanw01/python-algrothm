# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def invertTree(self, root):
        """
        input: TreeNode root
        return: TreeNode
        """
        # write your solution here
        if root is None:
            return
        left = root.left
        right = root.right
        root.left = right
        root.right = left
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root


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
print(s.invertTree(t1))
