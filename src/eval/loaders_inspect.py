from src.eval.loaders import load_msc
from datasets import load_dataset


def inspect_datasets():
    print("--- Inspecting LongMemEval ---")
    try:
        # Load only the first 5 examples to inspect structure
        lme = load_dataset(
            "xiaowu0162/longmemeval-cleaned",
            split="longmemeval_s_cleaned",
            streaming=True,
        )
        print("Dataset loaded in streaming mode.")
        for i, example in enumerate(lme):
            print(f"Sample {i}: {example.keys()}")
            if i == 0:
                print(f"Full Sample 0: {example}")
            if i >= 2:
                break
    except Exception as e:
        print(f"Error loading LongMemEval: {e}")

    print("\n--- Inspecting MSC ---")
    try:
        msc = load_msc(split="validation")
        print(f"Size: {len(msc)}")
        print(f"Sample: {msc[0]}")
    except Exception as e:
        print(f"Error loading MSC: {e}")


if __name__ == "__main__":
    inspect_datasets()
