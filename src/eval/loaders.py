from datasets import load_dataset


def load_msc(split="validation"):
    """
    Loads the Multi-Session Chat dataset.
    """
    dataset = load_dataset("facebook/msc", split=split)
    return dataset


def load_longmemeval(split="longmemeval_s_cleaned"):
    """
    Loads the LongMemEval dataset.
    Available splits: 'longmemeval_s_cleaned', 'longmemeval_m_cleaned', 'longmemeval_oracle'
    """
    try:
        dataset = load_dataset("xiaowu0162/longmemeval-cleaned", split=split)
    except Exception as e:
        print(f"Failed to load cleaned version, trying original: {e}")
        try:
            dataset = load_dataset("xiaowu0162/longmemeval", split=split)
        except Exception as e2:
            print(f"Failed to load original version: {e2}")
            raise e
    return dataset

    return dataset


class MockLongMemEvalDataset:
    def __init__(self, size=10):
        self.size = size
        self.data = []
        for i in range(size):
            self.data.append(
                {
                    "question": f"What is the capital of country_{i}?",
                    "answer": f"Capital_{i}",
                    "history": [
                        f"I live in country_{i}.",
                        f"The capital of country_{i} is Capital_{i}.",
                    ],
                    "question_id": f"q_{i}",
                }
            )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


if __name__ == "__main__":
    # Test loading
    try:
        msc = load_msc()
        print(f"MSC loaded: {len(msc)} examples")
    except Exception as e:
        print(f"MSC Load Failed: {e}")

    try:
        lme = load_longmemeval()
        print(f"LongMemEval loaded: {len(lme)} examples")
    except Exception as e:
        print(f"LongMemEval Load Failed: {e}. Using Mock.")
        lme = MockLongMemEvalDataset()
        print(f"Mock LongMemEval loaded: {len(lme)} examples")
