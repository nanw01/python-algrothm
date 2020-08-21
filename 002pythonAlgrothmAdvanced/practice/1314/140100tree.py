# 1. 构建树    
# 我们先构建一棵简单的树：


class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
 
a = TreeNode(1)
b = TreeNode(2)
c = TreeNode(3)
d = TreeNode(4)
e = TreeNode(5)
f = TreeNode(6)
g = TreeNode(7)
 
a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
c.right = g

# 2. 前序遍历
# 根节点->左子树->右子树

# 先序打印二叉树（递归）
def preOrderTraverse(node):
    if not node:
        return None
    print(node.val)
    preOrderTraverse(node.left)
    preOrderTraverse(node.right)


# 先序打印二叉树（非递归）
def preOrderTravese1(node):
    stack = [node]
    while len(stack) > 0:
        print(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
        node = stack.pop()



# 3. 中序遍历
# 左子树->根节点->右子树

# 中序打印二叉树（递归）
def inOrderTraverse(node):
    if node is None:
        return None
    inOrderTraverse(node.left)
    print(node.val)
    inOrderTraverse(node.right)

# 中序打印二叉树（非递归）
def inOrderTraverse1(node):
    stack = []
    pos = node
    while pos or len(stack) > 0:
        if pos:
            stack.append(pos)
            pos = pos.left
        else:
            pos = stack.pop()
            print(pos.val)
            pos = pos.right


# 4. 后序遍历
# 左子树->右子树->根节点

# 后序打印二叉树（递归）
def postOrderTraverse(node):
    if node is None:
        return None
    postOrderTraverse(node.left)
    postOrderTraverse(node.right)
    print(node.val)


# 后序打印二叉树（非递归）
# 使用两个栈结构
# 第一个栈进栈顺序：左节点->右节点->跟节点
# 第一个栈弹出顺序： 跟节点->右节点->左节点(先序遍历栈弹出顺序：跟->左->右)
# 第二个栈存储为第一个栈的每个弹出依次进栈
# 最后第二个栈依次出栈
def postOrderTraverse1(node):
    stack = [node]
    stack2 = []
    while len(stack) > 0:
        node = stack.pop()
        stack2.append(node)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    while len(stack2) > 0:
        print(stack2.pop().val)

# 5. 层次遍历
# 逐层遍历

def layerTraverse(node):
    
    if not node:
        return None
 
    queue = []  
    queue.append(node)
    while len(queue) > 0:
        tmp = queue.pop(0)
        print(tmp.val)
        if tmp.left:
            queue.append(tmp.left)
        if tmp.right:
            queue.append(tmp.right)


print('前序',preOrderTraverse(a))
print('中序',inOrderTraverse(a))
print('后序',postOrderTraverse(a))
print('层级',layerTraverse(a))