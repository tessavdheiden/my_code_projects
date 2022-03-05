import re
import itertools
from typing import List

from node import Node


class GeneralNode(Node):
    def __init__(self, value):
        super().__init__(value)
        self.children = []

    @property
    def levels(self) -> List[List["Node"]]:
        current_nodes = [self]
        levels = []

        while len(current_nodes) > 0:
            next_nodes = []

            for node in current_nodes:
              for child in node.children:
                if child is not None:
                  next_nodes.append(child)

            levels.append(current_nodes)
            current_nodes = next_nodes

        return levels
    def pprint(self, index: bool = False, delimiter: str = "-") -> None:
      tree = []

      for level in self.levels[::-1]:
        generation = []
        for node in level:
          generation.append(node.val)
        tree.append(generation)
      pretty_print(tree, n_children=len(self.children))


def create_pipes_from_numbers_in_line(line):
    pipes = " " * len(line)
    matches = re.finditer(r'[ _]\d', line)
    indeces = [match.start() + 1 for match in matches]
    for index in indeces:
        pipes = pipes[:index] + "|" + pipes[index + 1:]
    return pipes


def replace_underscores_between_pipes(line, n_children):
    matches = re.finditer(r'\|', line)
    spans = [match.span() for match in matches]
    spans_between_pipes = [(t1[1], t2[0]) for t1, t2 in zip(spans[:-1], spans[1:])]
    are_children = [True if i % n_children != (n_children - 1) else False for i in range(len(spans_between_pipes))]
    spans_between_pipes = list(itertools.compress(spans_between_pipes, are_children))

    for i, (start, end) in enumerate(spans_between_pipes):
        line = line[:start] + "_" * (end - start) + line[end:]
    return line


def create_next_generation_numbers(line, next_line):
    numbers = " " * len(line)
    matches = re.finditer(r' +', line)
    spans = [match.span() for match in matches]
    # spans = [(0,0),*spans]
    spans = [(t1[1], t2[0]) for t1, t2 in zip(spans[:-1], spans[1:])]
    for i, (start, end) in enumerate(spans):
        char_number = str(next_line[i])
        number_length = len(char_number)
        span_length = end - start
        l_box = (span_length - number_length) // 2
        numbers = numbers[:start] + " " * l_box + char_number + " " * l_box + numbers[end:]
    return numbers


def pretty_print(tree, n_children):
    line = " " + " ".join([str(i) for i in tree[0]]) + " "
    for j, generation in enumerate(tree[1:]):
        pipes = create_pipes_from_numbers_in_line(line)

        pipes = replace_underscores_between_pipes(pipes, n_children)

        print(line)
        print(pipes)

        parent_generation = tree[j + 1]

        parents = create_next_generation_numbers(pipes, parent_generation)

        line = parents

    print(parents)