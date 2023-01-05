from typing import Set, Iterable, Iterator, Optional, NewType
from itertools import product
import networkx as nx  # type: ignore

from dbt.exceptions import InternalException

UniqueId = NewType("UniqueId", str)


class Graph:
    """A wrapper around the networkx graph that understands SelectionCriteria
    and how they interact with the graph.
    """

    def __init__(self, graph):
        self.graph = graph

    def nodes(self) -> Set[UniqueId]:
        return set(self.graph.nodes())

    def edges(self):
        return self.graph.edges()

    def __iter__(self) -> Iterator[UniqueId]:
        return iter(self.graph.nodes())

    def ancestors(self, node: UniqueId, max_depth: Optional[int] = None) -> Set[UniqueId]:
        """Returns all nodes having a path to `node` in `graph`"""
        if not self.graph.has_node(node):
            raise InternalException(f"Node {node} not found in the graph!")
        return {
            child
            for _, child in nx.bfs_edges(self.graph, node, reverse=True, depth_limit=max_depth)
        }

    def descendants(self, node: UniqueId, max_depth: Optional[int] = None) -> Set[UniqueId]:
        """Returns all nodes reachable from `node` in `graph`"""
        if not self.graph.has_node(node):
            raise InternalException(f"Node {node} not found in the graph!")
        return {child for _, child in nx.bfs_edges(self.graph, node, depth_limit=max_depth)}

    def select_childrens_parents(self, selected: Set[UniqueId]) -> Set[UniqueId]:
        ancestors_for = self.select_children(selected) | selected
        return self.select_parents(ancestors_for) | ancestors_for

    def select_children(
        self, selected: Set[UniqueId], max_depth: Optional[int] = None
    ) -> Set[UniqueId]:
        descendants: Set[UniqueId] = set()
        for node in selected:
            descendants.update(self.descendants(node, max_depth))
        return descendants

    def select_parents(
        self, selected: Set[UniqueId], max_depth: Optional[int] = None
    ) -> Set[UniqueId]:
        ancestors: Set[UniqueId] = set()
        for node in selected:
            ancestors.update(self.ancestors(node, max_depth))
        return ancestors

    def select_successors(self, selected: Set[UniqueId]) -> Set[UniqueId]:
        successors: Set[UniqueId] = set()
        for node in selected:
            successors.update(self.graph.successors(node))
        return successors

    def trim_unvisited_nodes(
        self, target_graph: nx.DiGraph, selected_nodes: Set[UniqueId]
    ) -> "Graph":
        """Method that modifies the graph by removing unnecessary nodes
        from graph without effecting selection."""

        all_nodes: Set[UniqueId] = set(target_graph.nodes)

        for node in selected_nodes:
            ancestors: Set[UniqueId] = set(nx.ancestors(target_graph, node))
            predecessors: Set[UniqueId] = set(nx.predecessor(target_graph, node))
            predecessors = predecessors.difference({node})
            visited_nodes = predecessors.union(ancestors).union(selected_nodes)
            unvisited_nodes = all_nodes.difference(visited_nodes)

            if not selected_nodes.intersection(predecessors):
                target_graph.remove_nodes_from(predecessors)
            if not selected_nodes.intersection(ancestors):
                target_graph.remove_nodes_from(ancestors)
            target_graph.remove_nodes_from(unvisited_nodes)

    def get_subset_graph(self, selected: Iterable[UniqueId]) -> "Graph":
        """Create and return a new graph that is a shallow copy of the graph,
        but with only the nodes in include_nodes. Transitive edges across
        removed nodes are preserved as explicit new edges.
        """

        new_graph = self.graph.copy()
        include_nodes = set(selected)

        self.trim_unvisited_nodes(new_graph, include_nodes)

        all_nodes = set(new_graph.nodes)
        nodes_to_remove = all_nodes - include_nodes

        for node in nodes_to_remove:
            possible_edges = product(new_graph.predecessors(node), new_graph.successors(node))
            non_cyclic_edges = [
                (source, target) for source, target in possible_edges if source != target
            ]
            new_graph.remove_node(node)
            new_graph.add_edges_from(non_cyclic_edges)

        for node in include_nodes:
            if node not in new_graph:
                raise ValueError(
                    "Couldn't find model '{}' -- does it exist or is it disabled?".format(node)
                )

        return Graph(new_graph)

    def subgraph(self, nodes: Iterable[UniqueId]) -> "Graph":
        return Graph(self.graph.subgraph(nodes))

    def get_dependent_nodes(self, node: UniqueId):
        return nx.descendants(self.graph, node)
