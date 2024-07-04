## Llama 3 (8B Version) Deep Dive

Llama 3, developed by Meta AI, represents a significant advancement in open-source language models. The 8B parameter version used in this project offers a balance between capability and efficiency.

### Architecture Insights:
- Based on the transformer architecture with improvements.
- Utilizes rotary positional embeddings (RoPE) for enhanced position awareness.
- Employs SwiGLU activation function: swish(x) * (Wx + b), where swish(x) = x * sigmoid(x).

### Mathematical Foundation:
1. Self-Attention Mechanism:
   Q = XW_Q, K = XW_K, V = XW_V
   Attention(Q, K, V) = softmax(QK^T / √d_k)V
   Where d_k is the dimension of the key vectors.

2. Feed-Forward Network:
   FFN(x) = W_2 * SwiGLU(W_1x + b_1) + b_2

3. Layer Normalization:
   LN(x) = α * (x - μ) / (σ + ε) + β
   Where μ is the mean, σ is the standard deviation, and α, β are learnable parameters.

### Interesting Facts:
- Trained on a diverse dataset of over 2 trillion tokens.
- Uses a context window of 4096 tokens, allowing for processing of longer sequences.
- Implements flash attention for faster and more memory-efficient attention computation.

### Performance Metrics:
- Achieves near state-of-the-art performance on various NLP benchmarks despite its relatively smaller size.
- Demonstrates strong few-shot learning capabilities, often matching larger models in zero-shot scenarios.

## Low-Rank Adaptation (LoRA) In-Depth

LoRA is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining model performance.

### Mathematical Formulation:
Instead of learning a large update matrix ΔW, LoRA decomposes it into two lower-rank matrices:
ΔW = BA, where B ∈ R^{d×r} and A ∈ R^{r×k}

The update rule becomes:
h = W_0x + BAx = W_0x + ΔWx
Where W_0 is the pre-trained weight matrix, and r is the rank (typically much smaller than d and k).

### Key Advantages:
1. Memory Efficiency: Reduces parameter count from d*k to r*(d+k).
2. Computational Efficiency: Reduces FLOPs for both training and inference.
3. Adaptability: Allows for task-specific fine-tuning without altering the base model.

### LoRA in Practice:
- Applied to specific layers: query, key, value, and output projections in self-attention, and up/down projections in feed-forward layers.
- Rank (r) is a hyperparameter: Higher r allows for more expressive adaptations but increases memory usage.
- Scaling factor α is used to control the magnitude of LoRA updates.

### Mathematical Insight:
The effectiveness of LoRA can be partly explained by the concept of "intrinsic dimension" in deep learning. Despite having millions of parameters, the functional space of a neural network often has a much lower intrinsic dimension, which LoRA exploits.

### Interesting Applications:
- Multi-task Learning: Different LoRA adapters can be trained for different tasks on the same base model.
- Continual Learning: LoRA adapters can be sequentially trained to adapt to new tasks without forgetting previous ones.

### Performance Comparisons:
- In many cases, LoRA with r=16 achieves 95%+ of the performance of full fine-tuning while updating less than 1% of the parameters.
- LoRA has been shown to outperform other parameter-efficient fine-tuning methods like adapter tuning and prefix tuning in various scenarios.
