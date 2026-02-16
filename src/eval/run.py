import wandb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from src.agent import create_agent_workflow, init_memtree, reset_memtree
from src.eval.loaders import load_longmemeval, MockLongMemEvalDataset


def run_eval():
    # Initialize WandB
    wandb.init(project="memtree-eval")

    # Setup Agent
    import os

    if os.environ.get("OPENAI_API_KEY"):
        embedding_model = OpenAIEmbeddings()
        llm = ChatOpenAI(model="gpt-4o")
        print("Using Real LLM")
    else:
        print("OPENAI_API_KEY not found. Using Mock LLM.")
        from tests.test_memtree import MockEmbeddings, MockLLM

        embedding_model = MockEmbeddings()
        llm = MockLLM()

    # Initialize MemTree globally for the agent
    memtree = init_memtree(embedding_model, llm)
    app = create_agent_workflow()

    # Load Dataset
    try:
        dataset = load_longmemeval(
            split="longmemeval_s_cleaned"
        )  # or validation if available
        # Check if dataset is empty or broken
        if len(dataset) == 0:
            raise ValueError("Empty dataset")
        print("Loaded Real LongMemEval dataset")
    except Exception as e:
        print(f"Failed to load real dataset: {e}. Using Mock Dataset.")
        dataset = MockLongMemEvalDataset()

    # Metrics
    total_correct = 0
    total_examples = 0

    # Loop through examples (limit for testing)
    for i, example in enumerate(dataset):
        if i >= 10:
            break  # Run 10 examples for now

        # LongMemEval structure might vary, need to inspect.
        # Assuming it has 'context' or 'history' and a 'question' + 'answer'.
        # Let's assume standard format for now and adjust after inspection.
        # Usually: 'history' (list of strings), 'question', 'answer'.

        # Populate memory with history
        # (Reset memory for each example? Or keep persistent?
        # Benchmark usually implies isolated evaluation or sequential.
        # LongMemEval tests "long-term memory", so probably populate history first.)

        # Reset memtree for each example to ensure isolation?
        # Or is it a single long conversation?
        # LongMemEval -> "500 questions embedded within... chat histories".
        # Likely separate sessions. I will reset.
        reset_memtree()
        memtree = init_memtree(embedding_model, llm)

        history = example.get("history", [])  # Adjust key based on dataset inspection
        question = example.get("question", "")  # Adjust key
        ground_truth = example.get("answer", "")  # Adjust key

        # Insert history
        for msg in history:
            memtree.insert(msg)

        # Run Agent on Question
        state = {"messages": [HumanMessage(content=question)]}
        result = app.invoke(state)
        ai_response = result["messages"][-1].content

        # Simple exact match or LLM judge?
        # For now, let's log input/output and use a simple containment check or exact match
        is_correct = ground_truth.lower() in ai_response.lower()

        if is_correct:
            total_correct += 1
        total_examples += 1

        wandb.log(
            {
                "question": question,
                "response": ai_response,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "example_id": i,
            }
        )

    accuracy = total_correct / total_examples if total_examples > 0 else 0
    wandb.log({"accuracy": accuracy})
    print(f"Evaluation Complete. Accuracy: {accuracy}")
    wandb.finish()


if __name__ == "__main__":
    run_eval()
