# Decodex : A GPT Decoder-Only Model from Scratch

## Overview
This project implements a **decoder-only GPT model** from scratch using PyTorch. The model is inspired by the **Transformer architecture** detailed in the "Attention Is All You Need" paper. It trains on character-level tokenization and can generate text that mimics the style of the training data.

## Features
- **Fully implemented from scratch**: No pre-built transformer modules used.
- **Character-level tokenization**: Works at the level of individual characters.
- **Multi-head self-attention mechanism**: Implements scalable self-attention.
- **Positional embeddings**: Maintains sequence order information.
- **Layer normalization and dropout**: Improves training stability.
- **Autoregressive text generation**: Generates text by predicting the next token iteratively.

## Model Architecture
The model follows a typical decoder-only Transformer design with:
- **Token and Positional Embeddings**
- **Multiple Transformer Blocks** (each consisting of:
  - Multi-Head Self-Attention
  - Feedforward Network
  - Layer Normalization)
- **Final Linear Layer for Token Prediction**

## Hyperparameters
| Parameter        | Value |
|-----------------|-------|
| Batch Size      | 16    |
| Block Size      | 32    |
| Embedding Size  | 64    |
| Heads          | 4     |
| Layers         | 4     |
| Learning Rate   | 1e-3  |
| Max Iterations  | 5000  |
| Dropout         | 0.0   |

## Dataset
The model is trained on **The Adventure of the Empty House** (a Sherlock Holmes story). The text is first converted into a vocabulary of unique characters, which are then mapped to integers for training.

## Installation
To run this project, ensure you have Python and PyTorch installed:
```bash
pip install torch
```
Clone the repository:
```bash
git clone https://github.com/your-username/Decodex.git
cd Decodex
```

## Usage
### Training the Model
Run the training script to train the model from scratch:
```bash
python train.py
```

### Generating Text
After training, you can generate text using:
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
```

## Results
The trained model generates text that mimics the style of the training data. Example output:
```
Sherlock Holmes stood at the door, his keen eyes scanning the room. "Watson," he said, "observe the details."
```

## Future Improvements
- Train on larger datasets for better fluency.
- Experiment with different tokenization techniques.
- Implement more advanced training optimizations.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's "Zero to Hero"](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## License
This project is open-source under the MIT License.

