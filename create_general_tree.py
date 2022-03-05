
from general_node import GeneralNode


def create_general_tree(depth=5, width=2, sparsity=0.5, reverse=False):
    root = GeneralNode(1)
    for n in range(1, width + 1):
        root.children.append(GeneralNode(root.value + n))

    curr = root.children[-1].value + 1
    l = 1
    while l <= depth:
        level = root.levels[l]
        for node in level:
            for val in range(curr, curr + width):
                node.children.append(GeneralNode(val))
            curr += width
        l += 1

    depth = 0
    for level in root.levels:
        depth += 1
        for node in level:
            if len(node.children) == 0:
                if reverse:
                    node.reward = 10
                else:
                    node.reward = depth

    return root


if __name__ == "__main__":
    tree_node = create_general_tree(depth=2, width=3)
    tree_node.pprint()
