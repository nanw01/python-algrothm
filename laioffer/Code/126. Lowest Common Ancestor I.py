# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return None
        elif root == p or root == q:
            return root
        lres = self.lowestCommonAncestor(root.left, p, q)
        rres = self.lowestCommonAncestor(root.right, p, q)

        if lres and rres:
            return root

        if lres:
            return lres
        return rres


t1 = TreeNode(0)
t2 = TreeNode(1)
t3 = TreeNode(2)
t4 = TreeNode(3)
t5 = TreeNode(4)
t6 = TreeNode(5)
t7 = TreeNode(6)

t1.left = t2
t1.right = t5
t2.left = t3
t2.right = t4
t5.left = t6
t5.right = t7


s = Solution()
print(s.lowestCommonAncestor(t1, 5, 6))
