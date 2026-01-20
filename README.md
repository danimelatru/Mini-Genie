# Mini-Genie: Unsupervised Generative World Model

![Dream Sequence](assets/dream_real_data.gif)
*Figure 1: The model "dreaming" future states purely from latent space (Generative Transformer), maintaining object permanence without access to the game engine.*

## 🧠 Technical Deep Dive

This project implements a **Latent Action Model (LAM)** inspired by DeepMind's *Genie* (2024), capable of learning the physics and controls of a Multi-Agent Reinforcement Learning environment entirely from unlabeled video.

### Architecture & Key Challenges solved

#### 1. The Tokenizer (VQ-VAE) with Spatial Focus
Standard VQ-VAEs failed to reconstruct the single-pixel "apples" in the *Harvest* environment due to the dominance of the black background.
* **Solution:** Implemented a custom **Spatial Weighted L1 Loss** that penalizes reconstruction errors on non-background pixels by a factor of 20x.
* **Result:** High-fidelity tokenization of sparse reward objects.

#### 2. The Dynamics Model (Transformer)
Transitioned from an MLP-based predictor to a **Causal Transformer** to capture temporal dependencies.
* **Challenge:** "Gray Soup" hallucination due to stochastic uncertainty in early training.
* **Solution:** Implemented **Greedy Decoding (Determininstic Sampling)** during inference to stabilize video generation, proving the model learned distinct concepts of "Object Permanence" (walls and resources persist over time).

#### 3. Latent Action Discovery
Used an Inverse Dynamics Model to cluster continuous video transitions into discrete latent actions.
* **Insight:** The model successfully disentangled "Movement" dynamics from "Interaction" dynamics (zapping/shooting), grouping them into distinct latent clusters unsupervised.

## 🛠️ Tech Stack
* **Core:** PyTorch, Gym (Legacy), NumPy.
* **Models:** VQ-VAE (Residual), Transformer (GPT-style Decoder).
* **Vis:** Matplotlib, Seaborn (Confusion Matrices).

---