
"""
Initial evaluation dataset for NoteWise evidence sufficiency.

Labels:
- direct_support
- partial_support
- related_only
- no_support
"""

EVALUATION_CASES = [
    {
        "question": "What is machine learning?",
        "expected_label": "direct_support",
    },
    {
        "question": "How does deep learning work?",
        "expected_label": "direct_support",
    },
    {
        "question": "What is predictive AI?",
        "expected_label": "direct_support",
    },
    {
        "question": "How can AI detect fraudulent transactions?",
        "expected_label": "direct_support",
    },
    {
        "question": "What is generative AI?",
        "expected_label": "direct_support",
    },
    {
        "question": "Who founded OpenAI?",
        "expected_label": "no_support",
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "expected_label": "direct_support",
    },
    {
        "question": "How does reinforcement learning use rewards?",
        "expected_label": "direct_support",
    },
    {
        "question": "What is backpropagation in neural networks?",
        "expected_label": "related_only",
    },
    {
        "question": "What architecture does GPT-4 use?",
        "expected_label": "no_support",
    },
    {
        "question": "What is the capital of France?",
        "expected_label": "no_support",
    },
    {
        "question": "How does photosynthesis work?",
        "expected_label": "no_support",
    },
    {
        "question": "Who invented the telephone?",
        "expected_label": "no_support",
    },
    {
        "question": "What causes earthquakes?",
        "expected_label": "no_support",
    },
    {
        "question": "How does the human digestive system work?",
        "expected_label": "no_support",
    },
]
