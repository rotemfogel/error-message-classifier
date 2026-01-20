import json
import os
from typing import Dict, List

from dotenv import load_dotenv

from error_classification import ErrorMessageClassifier

_ERRORS_FILE = "errors.json"
_MODEL_CATEGORIES = "model_categories.json"
_MESSAGE_LABELS = "model_messages_and_labels.json"
_CLASSIFIER_MODEL = "error_classifier_model.pkl"
_SAMPLES_FILE = "sample_errors.json"


if __name__ == "__main__":
    load_dotenv()
    with open(_ERRORS_FILE, "r") as f:
        raw_errors = json.load(f)
    sample_errors = list(set([r["error_message"] for r in raw_errors]))

    # Initialize classifier
    classifier = ErrorMessageClassifier()

    categories: Dict[str, str] = dict()
    if os.path.exists(_MODEL_CATEGORIES):
        with open(_MODEL_CATEGORIES, "r") as f:
            categories = json.load(f)
    else:
        print("=" * 60)
        print("Step 1: Discovering categories from error messages...")
        print("=" * 60)
        categories = classifier.discover_categories_with_llm(sample_errors)
        with open(_MODEL_CATEGORIES, "w") as f:
            json.dump(categories, f)

    messages: List[str] = []
    labels: List[str] = []
    if os.path.exists(_MESSAGE_LABELS):
        with open(_MESSAGE_LABELS, "r") as f:
            messages_and_labels = json.load(f)
            messages = messages_and_labels["messages"]
            labels = messages_and_labels["labels"]
    else:
        print("\n" + "=" * 60)
        print("Step 2: Creating training set with LLM...")
        print("=" * 60)
        messages, labels = classifier.create_training_set_with_llm(sample_errors)
        with open(_MESSAGE_LABELS, "w") as f:
            json.dump({"messages": messages, "labels": labels}, f)

    if os.path.exists(_CLASSIFIER_MODEL):
        with open(_CLASSIFIER_MODEL, "rb") as f:
            classifier.load_model(_CLASSIFIER_MODEL)
    else:
        print("\n" + "=" * 60)
        print("Step 3: Training ML model for fast inference...")
        print("=" * 60)
        classifier.train_ml_model(messages, labels)
        # Save the model
        classifier.save_model(_CLASSIFIER_MODEL)

    print("\n" + "=" * 60)
    print("Step 4: Testing classification...")
    print("=" * 60)

    if os.path.exists(_SAMPLES_FILE):
        with open(_SAMPLES_FILE, "r") as f:
            test_errors = json.load(f)
    else:
        test_errors = sample_errors

    print("\nUsing fast ML model:")
    results = classifier.batch_predict(test_errors, use_llm=False)
    print(
        results.groupby(["category"])
        .agg(
            totals=("category", "count"),
            confidence=("confidence", "max"),
        )
        .sort_values(by=["confidence", "totals"], ascending=[False, False])
    )

    # Uncomment to use LLM for classification (slower but potentially more accurate)
    # print("\nUsing LLM (slower, more accurate):")
    # results_llm = classifier.batch_predict(test_errors[:2], use_llm=True)
    # print(results_llm.to_string(index=False))
