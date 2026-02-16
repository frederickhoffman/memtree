from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class MemNode(BaseModel):
    """
    Represents a node in the dynamic tree memory.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str
    embedding: Optional[List[float]] = None
    parent_id: Optional[UUID] = None
    children_ids: List[UUID] = Field(default_factory=list)
    depth: int = 0
    summary: Optional[
        str
    ] = None  # Can be used for aggregated content if different from 'content'

    def __repr__(self):
        return f"MemNode(id={self.id}, depth={self.depth}, content='{self.content[:20]}...', children={len(self.children_ids)})"
