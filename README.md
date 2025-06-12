# Decodex: A GPT Decoder-Only Model from Scratch

A minimal yet complete implementation of a decoder-only GPT model built entirely from scratch using PyTorch. This project demonstrates the core concepts of the Transformer architecture through character-level text generation.

## 🚀 Features

- **Pure PyTorch implementation** - No pre-built transformer modules, every component built from scratch
- **Character-level tokenization** - Simple yet effective approach for text processing
- **Multi-head self-attention** - Scalable attention mechanism with configurable heads
- **Positional embeddings** - Maintains positional information in sequences
- **Regularization techniques** - Layer normalization and dropout for training stability
- **Autoregressive generation** - Iterative text generation with configurable length

## 🏗️ Architecture

The model implements a classic decoder-only Transformer design:

```
Input Text → Token Embeddings + Positional Embeddings
           ↓
    Transformer Blocks (×4)
    ├── Multi-Head Self-Attention
    ├── Layer Normalization
    ├── Feed-Forward Network
    └── Residual Connections
           ↓
    Linear Projection → Output Probabilities
```

Each transformer block contains:
- Multi-head self-attention mechanism
- Position-wise feed-forward network
- Layer normalization and residual connections

## ⚙️ Configuration

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Batch Size | 16 | Training batch size |
| Block Size | 32 | Maximum sequence length |
| Embedding Dimension | 64 | Token embedding size |
| Attention Heads | 4 | Number of attention heads |
| Transformer Layers | 4 | Number of transformer blocks |
| Learning Rate | 1e-3 | Adam optimizer learning rate |
| Training Steps | 5000 | Maximum training iterations |
| Dropout Rate | 0.0 | Dropout probability |

## 📚 Training Data

The model trains on classic literature for character-level text generation:
- **The Adventure of the Empty House** (Sherlock Holmes)
- **Tiny Shakespeare** dataset

Text is tokenized at the character level, creating a vocabulary from unique characters in the training corpus.

## 🛠️ Installation

**Prerequisites:**
- Python 3.7+
- PyTorch

**Setup:**
```bash
# Install PyTorch
pip install torch

# Clone the repository
git clone https://github.com/sourize/Decodex.git
cd Decodex
```

## 🚀 Usage

### Training
Start training the model from scratch:
```bash
python decodex.py
```

### Text Generation
Generate text after training:
```python
import torch

# Initialize generation context
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate text
generated_text = model.generate(context, max_new_tokens=2000)
output = decode(generated_text[0].tolist())
print(output)
```

## 📊 Sample Output

The trained model produces text in the style of its training data:

```
Sherlock Holmes stood at the door, his keen eyes scanning the room.
"Watson," he said, "observe the details. The game is afoot!"
```

## 🔮 Future Enhancements

- [ ] **Scaled training** - Train on larger, diverse datasets
- [ ] **Advanced tokenization** - Implement BPE or SentencePiece
- [ ] **Optimization improvements** - Add learning rate scheduling and gradient clipping
- [ ] **Model scaling** - Experiment with larger architectures
- [ ] **Evaluation metrics** - Add perplexity and other generation quality metrics
- [ ] **Interactive demo** - Web interface for text generation

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- [Andrej Karpathy's "Let's Build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Educational implementation guide

## 📄 License

MIT License - feel free to use this code for learning and experimentation.

---

⭐ **Star this repo** if you found it helpful for understanding Transformers from scratch!
