from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from encoder import Encoder

class GPT2:
    """
    A NumPy-based implementation of the GPT-2 language model.

    This class can load pre-trained GPT-2 model weights and generate text
    based on a given prompt. The core logic is implemented in the `forward`
    method, which processes input tokens through the transformer blocks.
    """
    # Define constants for special tokens and model parameters
    EOS_TOKEN = 50256  # End Of Sentence token ID
    DEFAULT_EPSILON = 1e-5

    def __init__(self, encoder: Encoder, hparams: Dict[str, Any], params: Dict[str, np.ndarray]):
        """
        Initializes the GPT-2 model with its configuration and weights.

        Args:
            encoder: An encoder object with `encode` and `decode` methods.
            hparams: A dictionary of hyperparameters (e.g., n_ctx, n_head).
            params: A dictionary containing the model's learned weights and parameters.
        """
        self.encoder = encoder
        self.n_ctx = hparams['n_ctx']
        self.n_head = hparams['n_head']
        
        # Explicitly load parameters by key for clarity and robustness
        self.blocks = params['blocks']
        self.ln_f = params['ln_f']
        self.wpe = params['wpe']
        self.wte = params['wte']

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Computes softmax activation to convert logits to probabilities."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """Computes the Gaussian Error Linear Unit (GELU) activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def _attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Computes the scaled dot-product attention.

        Args:
            q: Query matrix.
            k: Key matrix.
            v: Value matrix.
            mask: Causal mask to prevent attention to future positions.

        Returns:
            The output of the attention mechanism.
        """

        scores = q @ k.T / np.sqrt(q.shape[-1]) + mask
        return GPT2._softmax(scores) @ v

    def _layer_norm(self, x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x: Input tensor.
            g: Gain (gamma) parameter.
            b: Bias (beta) parameter.

        Returns:
            The normalized tensor.
        """

        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        # Normalize, then apply learnable gain and bias
        return g * (x - mean) / np.sqrt(variance + self.DEFAULT_EPSILON) + b

    def _multi_head_attention(self, x: np.ndarray, block: Dict[str, Any]) -> np.ndarray:
        """
        Performs the entire multi-head attention operation.

        This includes projection to QKV, splitting into heads, applying attention,
        concatenating heads, and the final output projection.
        
        Args:
            x: Input tensor after the first layer normalization.
            block: The current transformer block's weights.

        Returns:
            The output tensor from the multi-head attention mechanism.
        """

        # 1. Project to Q, K, V
        attn_weights = block['attn']['c_attn']['w']
        attn_bias = block['attn']['c_attn']['b']
        x_projected = x @ attn_weights + attn_bias

        # 2. Split into Q, K, V tensors
        qkv = np.split(x_projected, 3, axis=-1)

        # 3. Split Q, K, V into multiple heads
        qkv_heads = [np.split(tensor, self.n_head, axis=-1) for tensor in qkv]
        
        # 4. Apply causal mask to prevent attending to future tokens (-1e10 is used over -np.inf for numerical purposes)
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
        
        # 5. Compute attention for each head
        out_heads = [self._attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
        
        # 6. Concatenate attention heads and project output
        heads_matrix = np.hstack(out_heads)
        proj_weights = block['attn']['c_proj']['w']
        proj_bias = block['attn']['c_proj']['b']
        return heads_matrix @ proj_weights + proj_bias
        
    def _feed_forward(self, x: np.ndarray, block: Dict[str, Any]) -> np.ndarray:
        """
        Performs the feed-forward network (MLP) part of the transformer block.

        Args:
            x: Input tensor after the second layer normalization.
            block: The current transformer block's weights.

        Returns:
            The output tensor from the feed-forward network.
        """
        # First linear transformation
        fc_weights = block['mlp']['c_fc']['w']
        fc_bias = block['mlp']['c_fc']['b']
        ffn_hidden = x @ fc_weights + fc_bias
        
        # GELU activation
        ffn_activated = self._gelu(ffn_hidden)
        
        # Second linear transformation (projection)
        proj_weights = block['mlp']['c_proj']['w']
        proj_bias = block['mlp']['c_proj']['b']
        return ffn_activated @ proj_weights + proj_bias

    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Performs a forward pass of the GPT-2 model.

        Args:
            tokens: A list of input token IDs.

        Returns:
            A probability distribution over the vocabulary for the next token.
        """
        # 1. Token Embeddings + Positional Embeddings
        x = self.wte[tokens] + self.wpe[range(len(tokens))]
        
        # 2. Pass through all transformer decoder blocks
        for block in self.blocks:
            # First sub-layer: Multi-Head Attention
            ln1_x = self._layer_norm(x, block['ln_1']['g'], block['ln_1']['b'])
            x += self._multi_head_attention(ln1_x, block)
            
            # Second sub-layer: Feed-Forward Network
            ln2_x = self._layer_norm(x, block['ln_2']['g'], block['ln_2']['b'])
            x += self._feed_forward(ln2_x, block)
        
        # 3. Final Layer Normalization
        x_normalized = self._layer_norm(x, self.ln_f['g'], self.ln_f['b'])
        
        # 4. Final projection to vocabulary and softmax
        logits = x_normalized @ self.wte.T
        return self._softmax(logits[-1])

    def generate(self, tokens: List[int], n_tokens_to_generate: int) -> List[int]:
        """
        Generates a sequence of tokens starting from an initial list.

        Args:
            tokens: The initial list of token IDs (the prompt).
            n_tokens_to_generate: The number of new tokens to generate.

        Returns:
            The complete list of token IDs (prompt + generated text).
        """
        for _ in tqdm(range(n_tokens_to_generate), "Generating tokens"):
            probs = self.forward(tokens)
            
            # Use the most likely next word.
            next_id = np.argmax(probs)

            # To sample, you can replace the last line with:
            # next_id = np.random.choice(len(probs), p=probs)
            
            # Stop if the end-of-sentence token is generated
            if next_id == self.EOS_TOKEN:
                break
                
            tokens.append(next_id)
            
        return tokens

    def prompt(self, n_tokens_to_generate: int = 40) -> str:
        """
        Interactively prompts the user for input and generates a response.

        Args:
            n_tokens_to_generate: The maximum number of tokens for the response.

        Returns:
            The generated response as a string.
        """
        while True:
            prompt_text = input("Prompt: ")
            encoded_prompt = self.encoder.encode(prompt_text)
            
            # Validate that the prompt and generation length fit within the context window
            if len(encoded_prompt) + n_tokens_to_generate < self.n_ctx:
                break
            
            print(
                f"Error: Prompt length ({len(encoded_prompt)}) + generation length "
                f"({n_tokens_to_generate}) exceeds the model's context window ({self.n_ctx})."
            )
            
        encoded_response = self.generate(encoded_prompt, n_tokens_to_generate)
        return self.encoder.decode(encoded_response)