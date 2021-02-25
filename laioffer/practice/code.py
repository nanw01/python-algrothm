Node findSceondLargest(Node root) {
    // If tree is null or is single node only, return null(no second largest)
    if (root == null | | (root.left == null & & root.right == null)) return null
    Node parent = null, child = root
    // find the right most child
    while (child.right != null) {
        parent = child
        child = child.right
    }
    // if the right most child has no left child, then it's parent is second largest
    if (child.left == null) return parent
    // otherwise, return left child's rightmost child as second largest
    child = child.left
    while (child.right != null) child = child.right
    return child
}
