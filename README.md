# 🚀 HypeLLM: Hypothetical LLM Data Augmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- 🎮 **Recipe-Based Augmentation**: Pre-built recipes for common augmentation patterns
- 🔄 **Multiple Strategies**: Infer patterns, add reasoning, generate questions, and more
- 🎯 **Async & Sync Support**: Choose between async or sync APIs based on your needs
- ⚡ **Flexible Implementation**: Swap between different LLM backends (instructor, dspy, etc.)

## 🛠️ Installation

```bash
pip install hypellm
rye add hypellm
poetry add hypellm
```

Note that out of the box, you'll also need to install `instructor` as a peer dependency of `hypellm`, as it is the default implementation.

## 🚀 Quick Start

Take a look at [recipes.py](src/hypellm/recipes.py) to learn what's available and how they work.

```python
# env vars:
# HYPELLM_MODEL=gpt-4o # LiteLLM model name
# HYPELLM_API_KEY=your_api_key

import hypellm

# Your training examples, simple strings or structured data
data = [
    # Example(inputs: dict | str, reasoning: list[str], outputs: dict | str)
    hypellm.Example(
        inputs="The patient presents with elevated troponin levels (0.8 ng/mL) and ST-segment depression, but no chest pain or dyspnea.",
        outputs="unstable_angina"
    ),
    hypellm.Example(
        inputs="Labs show WBC 15k/μL with 80% neutrophils, fever 39.2°C, and consolidation in right lower lobe on chest X-ray.",
        outputs="bacterial_pneumonia"
    )
]

# Choose your implementation
hypellm.settings.impl_name = "instructor"  # or your custom impl

# Use different recipes
async def augment_examples():
    # Infer a prompt from examples
    prompt = await hypellm.recipes.inferred(data)
    print(f"Intent: {prompt.intent}")
    print(f"Do's: {prompt.dos}")
    print(f"Don'ts: {prompt.donts}")
    print(f"Reasoning Steps: {prompt.reasoning_steps}")
    print(f"Examples: {prompt.examples}")

    # Add reasoning steps to examples
    prompt, results = await hypellm.recipes.reasoned(data, prompt=prompt)
    for result in results:
        print(f"Q: {result.inputs}")
        print(f"Reasoning: {result.reasoning}")
        print(f"A: {result.outputs}")

    # Generate questions from different angles
    questions = await hypellm.recipes.questions(data)
    for question, data_that_answers_question in questions.items():
        print(f"{question}: {data_that_answers_question}")

    # Invert input/output pairs
    prompt, inverted = await hypellm.recipes.inverted(data)
    print(f"Inverted prompt: {prompt}")
    for result in inverted:
        print(f"Original: {result.outputs} -> [{result.reasoning}] -> {result.inputs}")
        print(f"Inverted: {result.inputs} -> [{result.reasoning}] -> {result.outputs}")
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with 🔥 by [Zenbase AI](https://zenbase.ai) - Empowering the next generation of LLM applications.

*Remember: Better data means better models!* 🎯
