# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a deep learning educational repository ("deeplearning_deepblue") organized as a structured course with 8 modules covering fundamental to advanced deep learning topics. Content is primarily in Chinese.

## Repository Structure

The repository is organized into 8 topic-based directories:

1. **1_neural_network/** - Neural networks fundamentals (MLE, logistic regression, perceptrons, backpropagation)
2. **2_convolutional_neural_network/** - CNN architectures and applications
3. **3_optimization_algorithms_and_parameter_tuning/** - Optimization methods and hyperparameter tuning
4. **4_pytorch_framework/** - PyTorch framework usage
5. **5_deeplearning_in_cv/** - Deep learning for computer vision
6. **6_recurrent_neural_network/** - RNN, LSTM, GRU architectures
7. **7_deeplearning_in_nlp/** - Deep learning for natural language processing
8. **8_attention_mechanism/** - Attention mechanisms and transformers

Each module directory contains:
- Markdown documentation (`.md`) with theory, formulas, and code examples
- PDF files (`.pdf`) with detailed course materials
- Python implementations of algorithms and examples

## Content Format

- **Documentation Language**: Chinese (Simplified)
- **Code Comments**: Mixed Chinese and English
- **Mathematical Notation**: LaTeX-style formatting in markdown
- **Code Style**: Educational/tutorial style with clear variable names and extensive comments

## Working with This Repository

### Adding New Module Content

When adding content to a module directory:
1. Create a descriptive markdown file (e.g., `Module_Name.md`) with theory and examples
2. Include Python code blocks demonstrating implementations from scratch
3. Reference PDF page numbers when content corresponds to course materials
4. Structure content with clear sections: 理论核心 (Theory), 辅助工具与代码 (Tools & Code), 课后作业解答 (Assignment Solutions)

### Code Examples

Code examples in this repository typically:
- Use NumPy for matrix operations and numerical computing
- Implement algorithms from scratch for educational purposes (not production-optimized)
- Include visualization with matplotlib when demonstrating concepts
- Follow a clear pattern: data setup → model definition → training loop → results

Example pattern from Neural_Network.md:
```python
import numpy as np

# Data and parameters
X = np.array([...])
y = np.array([...])

# Activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

# Initialize weights
W1 = np.random.uniform(-1, 1, (input_size, hidden_size))

# Training loop with forward and backward propagation
for i in range(epochs):
    # Forward pass
    # Backward pass
    # Weight updates
```

### Mathematical Content

When working with mathematical formulas:
- Preserve LaTeX-style notation in markdown
- Keep Chinese terminology alongside mathematical symbols
- Reference standard ML/DL notation conventions (e.g., θ for parameters, δ for error terms)

## Development Environment

- **Primary Language**: Python 3
- **Core Dependencies**: NumPy, matplotlib (inferred from code examples)
- **Additional Libraries**: May include PyTorch, TensorFlow (for later modules)

No build system, test framework, or linting configuration is currently present in the repository.
