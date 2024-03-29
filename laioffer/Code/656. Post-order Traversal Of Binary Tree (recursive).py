# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def postOrder(self, root):
        """
        input: TreeNode root
        return: Integer[]
        """
        # write your solution here
        res = []
        self._postOrder(root, res)
        return res

    def _postOrder(self, root, res):

        if root is None:
            return

        self._postOrder(root.left, res)
        self._postOrder(root.right, res)
        res.append(root.val)

        return


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
print(s.postOrder(t1))
