class Solution(object):
    def findSecond(self, node):
        if node is None:
            return None
        if not node.right:
            return None
        self.second = node.val

        self._findSecond(node)
        return self.second

    def _findSecond(self, node):
        if node.right:
            self.second = node.val
            self._findSecond(node.right)

        if not node.left:
            self.second = node.val
            return

        self._findSecond(node.left)
        self.second = node.val

# timeO(h)
# space(h)
