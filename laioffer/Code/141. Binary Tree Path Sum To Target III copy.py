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
        self.ret = 0
        hash = dict()
        self._exist(root, target, hash, 0)
        return self.ret > 0

    def _exist(self, node, target, hash, curSum):
        if node is None:
            return
        curSum = curSum + node.val
        if curSum == target:
            self.ret += 1
        if (curSum - target) in hash:
            self.ret += hash[(curSum-target)]
        if curSum in hash:
            hash[curSum] += 1
        else:
            hash[curSum] = 1

        self._exist(node.left, target, hash, curSum)
        self._exist(node.left, target, hash, curSum)

        hash[curSum] -= 1
        curSum = curSum - node.val

        return


node = TreeNode(5, TreeNode(2), TreeNode(
    11, TreeNode(6, TreeNode(3)), TreeNode(14)))
print(Solution().exist(node, 17))
