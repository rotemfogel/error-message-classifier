"""
Error Message Classification System
Uses OpenAI LLM for intelligent categorization and traditional ML for efficient classification
"""

import json
import os
import pickle
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

_MODEL = "gpt-5.2"


class ErrorMessageClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, api_key: str = None):
        """
        Initialize the classifier with OpenAI API key

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.categories = []
        self.category_descriptions = {}

    def discover_categories_with_llm(
        self, error_messages: List[str], num_samples: int = 200
    ) -> Dict[str, str]:
        """
        Use OpenAI to discover natural categories from error messages

        Args:
            error_messages: List of error message strings
            num_samples: Number of samples to analyze

        Returns:
            Dictionary mapping category names to descriptions
        """
        # Sample messages for analysis
        sample_messages = np.random.choice(
            error_messages, min(num_samples, len(error_messages)), replace=False
        ).tolist()

        prompt = f"""Analyze these application error messages and identify 5-8 distinct categories.
For each category, provide:
1. A clear category name (e.g., "Database Connection", "Authentication", "Network Timeout")
2. A brief description of what errors belong in this category

Error messages:
{json.dumps(sample_messages[:30], indent=2)}

Respond ONLY with valid JSON in this format:
{{
  "categories": {{
    "CategoryName1": "Description of category 1",
    "CategoryName2": "Description of category 2"
  }}
}}"""

        try:
            response = self.client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing application errors and creating taxonomies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)
            self.category_descriptions = result["categories"]
            self.categories = list(self.category_descriptions.keys())

            print(f"Discovered {len(self.categories)} categories:")
            for cat, desc in self.category_descriptions.items():
                print(f"  - {cat}: {desc}")

            return self.category_descriptions

        except Exception as e:
            print(f"Error discovering categories: {e}")
            # Fallback to default categories
            raise e

    def classify_with_llm(self, error_message: str) -> Tuple[str, float]:
        """
        Classify a single error message using OpenAI

        Args:
            error_message: The error message to classify

        Returns:
            Tuple of (category, confidence)
        """
        categories_str = "\n".join(
            [f"- {cat}: {desc}" for cat, desc in self.category_descriptions.items()]
        )

        prompt = f"""Classify this error message into ONE of these categories:

{categories_str}

Error message: "{error_message}"

Respond ONLY with valid JSON:
{{
  "category": "CategoryName",
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at classifying application errors.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            result = json.loads(response.choices[0].message.content)
            return result["category"], result["confidence"]

        except Exception as e:
            print(f"Error classifying with LLM: {e}")
            return "Unknown Error", 0.5

    def create_training_set_with_llm(
        self, error_messages: List[str], sample_size: int = 100
    ) -> Tuple[List[str], List[str]]:
        """
        Use LLM to create labeled training data

        Args:
            error_messages: List of error messages
            sample_size: Number of messages to label

        Returns:
            Tuple of (messages, labels)
        """
        print(f"Creating training set with {sample_size} samples using LLM...")

        sample_messages = np.random.choice(
            error_messages, min(sample_size, len(error_messages)), replace=False
        ).tolist()

        labels = []
        for i, msg in enumerate(sample_messages):
            if i % 10 == 0:
                print(f"  Labeled {i}/{len(sample_messages)} messages...")

            category, _ = self.classify_with_llm(msg)
            labels.append(category)

        print(f"Training set created. Distribution:")
        dist = Counter(labels)
        for cat, count in dist.most_common():
            print(f"  {cat}: {count}")

        return sample_messages, labels

    def train_ml_model(self, messages: List[str], labels: List[str]):
        """
        Train a traditional ML model for fast inference

        Args:
            messages: Training messages
            labels: Training labels
        """
        print("Training ML model...")

        # Vectorize text
        X = self.vectorizer.fit_transform(messages)
        y = np.array(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))

    def predict(self, error_message: str, use_llm: bool = False) -> Tuple[str, float]:
        """
        Classify an error message

        Args:
            error_message: The error to classify
            use_llm: Whether to use LLM (slower but more accurate) or ML model (faster)

        Returns:
            Tuple of (category, confidence)
        """
        if use_llm and self.client:
            return self.classify_with_llm(error_message)
        else:
            # Use trained ML model
            X = self.vectorizer.transform([error_message])
            proba = self.classifier.predict_proba(X)[0]
            category_idx = np.argmax(proba)
            category = self.classifier.classes_[category_idx]
            confidence = proba[category_idx]
            return category, confidence

    def batch_predict(
        self, error_messages: List[str], use_llm: bool = False
    ) -> pd.DataFrame:
        """
        Classify multiple error messages

        Args:
            error_messages: List of errors to classify
            use_llm: Whether to use LLM or ML model

        Returns:
            DataFrame with messages, categories, and confidence scores
        """
        results = []
        for msg in error_messages:
            category, confidence = self.predict(msg, use_llm=use_llm)
            results.append(
                {"error_message": msg, "category": category, "confidence": confidence}
            )

        return pd.DataFrame(results)

    def save_model(self, filepath: str):
        """Save the trained model and vectorizer"""
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "categories": self.categories,
            "category_descriptions": self.category_descriptions,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        self.categories = model_data["categories"]
        self.category_descriptions = model_data["category_descriptions"]
        print(f"Model loaded from {filepath}")
