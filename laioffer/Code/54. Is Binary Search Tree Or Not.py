# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def isBST(self, root):
        """
        input: TreeNode root
        return: boolean
        """
        # write your solution here

        lst = self._inorder(root)
        return lst == sorted(lst) and len(lst) == len(set(lst))

    def _inorder(self, node):
        if not node:
            return []
        return self._inorder(node.left)+[node.val]+self._inorder(node.right)


t1 = TreeNode(0)
t2 = TreeNode(1)
t3 = TreeNode(2)
t4 = TreeNode(3)
t5 = TreeNode(4)
t6 = TreeNode(5)
t7 = TreeNode(6)

# t1.left = t2
# t1.right = t5
# t2.left = t3
# t2.right = t4
# t5.left = t6
# t5.right = t7

t2.left = t1
t2.right = t3

t4.left = t2
t4.right = t6

t6.left = t5
t6.right = t7


s = Solution()
print(s.isBST(t1))
