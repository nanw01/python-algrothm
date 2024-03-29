# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


class Solution(object):
    def exist(self, root, target):
        """
        input: TreeNode root, int target
        return: boolean
        """
        # write your solution here
        hash = dict()
        self.ret = 0
        self._exist(root, target, hash, 0)

        return self.ret > 0

    def _exist(self, node, target, hash, cur):
        if node is None:
            return

        cur = node.val + cur
        if cur == target:
            self.ret += 1
        if (cur - target) in hash:
            self.ret += hash[(cur-target)]
        if cur in hash:
            hash[cur] += 1
        else:
            hash[cur] = 1

        self._exist(node.left, target, hash, cur)
        self._exist(node.right, target, hash, cur)

        hash[cur] -= 1
        cur = cur - node.val

        return


node = TreeNode(5, TreeNode(2), TreeNode(
    11, TreeNode(6, TreeNode(3)), TreeNode(14)))
print(Solution().exist(node, 17))
