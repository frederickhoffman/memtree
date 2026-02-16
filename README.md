# MemTree: Dynamic Tree Memory for LLMs

An implementation of the hierarchical memory system described in the paper **"From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs"**.

MemTree organizes memory into a dynamic tree structure where:
- **Leaf nodes** represent raw, detailed observations (e.g., specific user messages).
- **Internal nodes** represent abstract schemas or summaries of their children.
- **Tree traversal** determines where new information belongs based on semantic similarity.
- **Incremental updates** ensure that upper-level schemas evolve as new details are added.

## Features

- **Dynamic Insertion**: Automatically routes new information to the most relevant semantic cluster or creates new branches.
- **Hierarchical Summarization**: Aggregates detailed memories into higher-level abstractions using LLMs.
- **Collapsed Retrieval**: Efficiently retrieves relevant information from any level of the hierarchy using flat semantic search.
- **LangGraph Integration**: state-of-the-art agent workflow integration.
- **WandB Monitoring**: Comprehensive experiment tracking and evaluation logging.

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/memtree.git
cd memtree

# Initialize and sync dependencies
uv sync
```

## Configuration

Set up your environment variables in a `.env` file or export them directly:

```bash
export OPENAI_API_KEY="sk-..."
export WANDB_API_KEY="wandb_..."
# Optional:
export LANGCHAIN_API_KEY="..."
export LANGCHAIN_TRACING_V2=true
```

## Usage

### Option 1: LangGraph UI

Run the agent interactively using the LangGraph Studio UI:

```bash
uv run langgraph dev
```

This will start a local server where you can chat with the agent and visualize the MemTree workflow.

### Option 2: Benchmarking Script

Run the evaluation script to test the agent against benchmarks (LongMemEval/MSC):

```bash
# Verify with mock data/fallback if dataset access is limited
uv run python evaluate.py
```

Results are logged to Weights & Biases under the project `memtree-eval`.

## Project Structure

- `src/memtree/`: Core implementation of `MemNode` and `MemTree` logic.
- `src/agent.py`: LangGraph agent definition.
- `src/eval/`: Dataset loaders and evaluation scripts.
- `tests/`: Unit tests ensuring core logic correctness.

## Pre-commit Hooks

Ensure code quality by running pre-commit hooks before pushing:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
