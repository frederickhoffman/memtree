import numpy as np
from typing import List, Optional, Dict
from uuid import UUID, uuid4
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .node import MemNode

AGGREGATE_PROMPT = """You will receive two pieces of information:
New Information is detailed, and Existing Information is a summary from {n_children} previous entries.
Your task is to merge these into a single, cohesive summary that highlights the most important insights.
- Focus on the key points from both inputs.
- Ensure the final summary combines the insights from both pieces of information.
- If the number of previous entries in Existing Information is accumulating (more than 2), focus on summarizing more concisely, only capturing the overarching theme, and getting more abstract in your summary.
Output the summary directly.

[New Information]
{new_content}
[Existing Information (from {n_children} previous entries)]
{current_content}

IMPORTANT! Don't output additional commentary, explanations, or unrelated information. Provide only the exact information or output requested.
[Output Summary]
"""


class MemTree:
    def __init__(
        self,
        embedding_model: Embeddings,
        llm: BaseChatModel,
        theta_base: float = 0.5,
        lambda_param: float = 1.1,
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.theta_base = theta_base
        self.lambda_param = lambda_param
        self.nodes: Dict[UUID, MemNode] = {}
        self.root_id = uuid4()
        # Initialize root node
        self.nodes[self.root_id] = MemNode(id=self.root_id, content="ROOT", depth=0)

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _compute_threshold(self, depth: int) -> float:
        # Threshold increases with depth, making it harder to go deeper unless very similar
        return self.theta_base * (self.lambda_param**depth)

    def insert(self, content: str):
        """
        Inserts new content into the memory tree.
        """
        embedding = self._get_embedding(content)
        new_node = MemNode(content=content, embedding=embedding)

        current_node_id = self.root_id

        # Traversal logic
        while True:
            current_node = self.nodes[current_node_id]
            children = [self.nodes[child_id] for child_id in current_node.children_ids]

            if not children:
                # No children, insert here
                self._add_child(current_node_id, new_node)
                break

            # Calculate similarities with all children
            similarities = []
            for child in children:
                if child.embedding:
                    sim = self._cosine_similarity(embedding, child.embedding)
                    similarities.append((sim, child.id))
                else:
                    similarities.append(
                        (-1.0, child.id)
                    )  # Should not happen for content nodes

            if not similarities:
                self._add_child(current_node_id, new_node)
                break

            best_sim, best_child_id = max(similarities, key=lambda x: x[0])
            threshold = self._compute_threshold(
                current_node.depth
            )  # Threshold for current level to go deeper?
            # Actually, paper: "If similarity > threshold, continue to child. Else insert as sibling (or child of current)"
            # Wait, if not similar enough to any child, it belongs to *current node* as a new child.

            if best_sim >= threshold:
                current_node_id = best_child_id
            else:
                # Not similar enough to any existing child, so it becomes a new child of the current node
                self._add_child(current_node_id, new_node)
                break

        # After insertion, update ancestors
        self._update_ancestors(new_node.parent_id, new_node.content)

    def _add_child(self, parent_id: UUID, child_node: MemNode):
        child_node.parent_id = parent_id
        child_node.depth = self.nodes[parent_id].depth + 1
        self.nodes[child_node.id] = child_node
        self.nodes[parent_id].children_ids.append(child_node.id)

    def _update_ancestors(self, start_node_id: Optional[UUID], new_content: str):
        """
        Updates summaries/embeddings of ancestor nodes up to the root incrementally.
        """
        current_id = start_node_id
        current_content_to_merge = new_content

        while current_id:
            node = self.nodes[current_id]
            if node.id == self.root_id:
                break  # Root doesn't need summary update in this version

            # Incremental update using AGGREGATE_PROMPT
            n_children = len(node.children_ids)
            # If node has no content yet (first child), just take the content
            if not node.content:
                node.content = current_content_to_merge
                node.embedding = self._get_embedding(current_content_to_merge)
                current_id = node.parent_id
                continue

            prompt = AGGREGATE_PROMPT.format(
                n_children=n_children,
                new_content=current_content_to_merge,
                current_content=node.content,
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content

            node.content = summary
            node.embedding = self._get_embedding(summary)

            # The summary of THIS node becomes the "new info" for the parent
            current_content_to_merge = summary
            current_id = node.parent_id

    def retrieve(self, query: str, k: int = 3) -> List[MemNode]:
        """
        Retrieves relevant nodes using collapsed tree retrieval (flat search).
        """
        query_embedding = self._get_embedding(query)

        candidates = []
        for node in self.nodes.values():
            if node.embedding:
                sim = self._cosine_similarity(query_embedding, node.embedding)
                candidates.append((sim, node))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in candidates[:k]]
