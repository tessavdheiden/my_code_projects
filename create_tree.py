
from node import Node


def create_tree(depth=5, sparsity=0.5, reverse=False):
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    l = 1
    curr = 4
    while l <= depth:
        level = root.levels[l]
        for node in level:
            node.left = Node(curr)
            curr += 1
            node.right = Node(curr)
            curr += 1
        l += 1

    return(root)


if __name__ == "__main__":
    tree_node = create_tree(1)
    tree_node.pprint()
