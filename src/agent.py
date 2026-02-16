from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .memtree.store import MemTree


# Global MemTree instance
_MEMTREE_INSTANCE = None


def init_memtree(embedding_model=None, llm=None):
    global _MEMTREE_INSTANCE
    if embedding_model is None:
        from langchain_openai import OpenAIEmbeddings

        embedding_model = OpenAIEmbeddings()
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o")
    _MEMTREE_INSTANCE = MemTree(embedding_model, llm)
    return _MEMTREE_INSTANCE


def get_memtree():
    global _MEMTREE_INSTANCE
    if _MEMTREE_INSTANCE is None:
        return init_memtree()
    return _MEMTREE_INSTANCE


def reset_memtree():
    global _MEMTREE_INSTANCE
    _MEMTREE_INSTANCE = None


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


def agent_node(state: AgentState):
    memtree = get_memtree()
    messages = state["messages"]
    last_message = messages[-1]

    # Retrieve context
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        context_nodes = memtree.retrieve(query, k=3)
        context_str = "\n".join([node.content for node in context_nodes])

        ANSWER_PROMPT = """Write a high-quality short answer for the given question using only the provided search results (some of which might be irrelevant).
[ Question ]
{query}
[ Search Results ]
{retrieved_content}
[ Output ]
"""
        final_prompt = ANSWER_PROMPT.format(query=query, retrieved_content=context_str)

        llm = ChatOpenAI(model="gpt-4o")  # Or configurable
        # Invoke as a single user message with the strict prompt
        response = llm.invoke([HumanMessage(content=final_prompt)])

        return {"messages": [response]}
    return {}


def save_memory_node(state: AgentState):
    memtree = get_memtree()
    messages = state["messages"]

    # Save the last interaction (User + AI)
    # We might want to save them separately or together.
    # For now, let's save the User's input and AI's response independently.

    # The 'messages' list is appended to. So the last two are likely User, AI.
    # But this node runs after 'agent_node'. So 'messages' has [..., User, AI].

    if len(messages) >= 2:
        user_msg = messages[-2]
        ai_msg = messages[-1]

        if isinstance(user_msg, HumanMessage):
            memtree.insert(f"User: {user_msg.content}")

        if isinstance(ai_msg, BaseMessage):  # AI message
            memtree.insert(f"AI: {ai_msg.content}")

    return {}


def create_agent_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("save_memory", save_memory_node)

    workflow.set_entry_point("agent")

    workflow.add_edge("agent", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()


graph = create_agent_workflow()
