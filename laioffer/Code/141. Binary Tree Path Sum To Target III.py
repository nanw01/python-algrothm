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
