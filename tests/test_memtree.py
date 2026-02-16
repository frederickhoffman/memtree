import unittest
from typing import List
from src.memtree.store import MemTree
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        if "apple" in text:
            return [1.0, 0.0, 0.0]
        if "banana" in text:
            return [0.9, 0.1, 0.0]  # Similar to apple
        if "car" in text:
            return [0.0, 1.0, 0.0]  # Different
        return [0.5, 0.5, 0.5]


class MockLLM(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="Summary of content"))
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "mock"


class TestMemTree(unittest.TestCase):
    def setUp(self):
        self.embeddings = MockEmbeddings()
        self.llm = MockLLM()
        self.memtree = MemTree(
            self.embeddings, self.llm, theta_base=0.5, lambda_param=1.1
        )

    def test_insert_root_children(self):
        self.memtree.insert("apple is a fruit")
        self.memtree.insert("car is a vehicle")

        # Should have 2 children under root (apple and car are distinct)
        root = self.memtree.nodes[self.memtree.root_id]
        self.assertEqual(len(root.children_ids), 2)

    def test_insert_hierarchy(self):
        self.memtree.insert("apple is a fruit")
        # "banana" is similar to "apple" (dot prod ~0.9), threshold at depth 0 is 0.5
        # So it should go UNDER "apple" (depth 1) or be a sibling?
        # Logic: if max_sim > threshold, go deep.
        # Here sim(apple, banana) ~0.9 > 0.5. So it enters "apple" node.
        # Inside "apple", no children yet. So "banana" becomes child of "apple".
        self.memtree.insert("banana is yellow")

        root = self.memtree.nodes[self.memtree.root_id]

        # Check that root has 1 child (the "apple" node, which is now a parent/summary)
        # Note: Depending on logic, if "apple" was replaced or updated.
        # "apple" node (id X) became parent.
        # Root -> [X]. X -> [Y].
        self.assertEqual(len(root.children_ids), 1)

        apple_node_id = root.children_ids[0]
        apple_node = self.memtree.nodes[apple_node_id]

        # "banana" should be a child of this node
        self.assertEqual(len(apple_node.children_ids), 1)
        banana_child_id = apple_node.children_ids[0]
        banana_child = self.memtree.nodes[banana_child_id]

        self.assertIn("banana", banana_child.content)
        self.assertEqual(banana_child.depth, 2)  # Root is 0, Apple is 1, Banana is 2

    def test_retrieve(self):
        self.memtree.insert("apple pie recipe")
        self.memtree.insert("driving a car")

        results = self.memtree.retrieve("apple", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("apple", results[0].content)


if __name__ == "__main__":
    unittest.main()
