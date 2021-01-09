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

        if not root:
            return []
        output, stack = [], [(root, 1)]

        while stack:
            node, count = stack.pop()
            if count == 3:
                output.append(node.val)
            if count == 1:
                stack.append((node, count + 1))
                if node.left:
                    stack.append((node.left, 1))
            if count == 2:
                stack.append((node, count + 1))
                if node.right:
                    stack.append((node.right, 1))
        return output


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
