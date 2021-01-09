# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def preOrder(self, root):
        """
        input: TreeNode root
        return: Integer[]
        """
        # write your solution here

        output = []
        if not root:
            return output
        stack = [(root, 1)]
        while stack:
            node, count = stack.pop()
            if count == 1:
                output.append(node.val)
                stack.append((node, count + 1))
                if node.left:
                    stack.append((node.left, 1))
            if count == 2:
                if node.right:
                    stack.append((node.right, 1))
        return output


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
print(s.preOrder(t1))
