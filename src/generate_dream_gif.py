import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import sys

# --- CONFIG ---
WINDOW_SIZE = 4 # Match the one in train_transformer_dynamics.py
VOCAB_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tokens"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
DREAM_STEPS = 50       
DREAM_ACTION_IDX = 0    
TEMPERATURE = 1.0       
TOP_K = 50              

def sample_top_k(logits, k, temperature):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    probs = F.softmax(out / temperature, dim=-1)
    flat_probs = probs.view(-1, VOCAB_SIZE)
    sample = torch.multinomial(flat_probs, 1)
    return sample.view(logits.shape[:-1])

def generate_gif(vqvae, world_model):
    print(f"🎬 Starting Dream Generation (Forcing Action {DREAM_ACTION_IDX})...")
    
    vqvae.eval()
    world_model.eval()
    
    # --- DREAM LOOP ---
    print(f"Dreaming {DREAM_STEPS} steps...")
    
    # We need a starting window of frames
    # Let's take the first WINDOW_SIZE frames from an episode
    episode_files = sorted(list(DATA_PATH.glob("*.npz")))
    if not episode_files:
        print("No episodes found to dream from.")
        return
        
    with np.load(episode_files[0]) as data:
        frames = data['frames'][:WINDOW_SIZE] # (W, 64, 64, 3)
        real_actions = data['action']

    # Encode initial window
    frames_torch = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
    z = vqvae.encoder(frames_torch)
    z = vqvae._pre_vq_conv(z)
    _, _, z_indices = vqvae.vq_layer(z)
    z_indices = z_indices.view(WINDOW_SIZE, 16, 16) # (W, 16, 16)
    
    current_window = z_indices.clone().unsqueeze(0) # (1, W, 16, 16)
    
    history_z = [z_indices[i].cpu().numpy() for i in range(WINDOW_SIZE)]

    for t in range(DREAM_STEPS):
        # 1. Choose Action (Use real action from trajectory if available, else random)
        if t + WINDOW_SIZE < len(real_actions):
            act = real_actions[t + WINDOW_SIZE]
        else:
            act = np.random.choice([1, 2, 3]) # Move/Turn
            
        act_onehot = torch.zeros(1, 8, device=DEVICE)
        act_onehot[0, act] = 1.0
        
        # 2. Predict next tokens from window
        with torch.no_grad():
            logits = world_model(current_window, act_onehot) # (1, 16, 16, VOCAB_SIZE)
            next_z = torch.argmax(logits, dim=-1) # (1, 16, 16)
        
        # 3. Update window (Slide)
        current_window = torch.cat([current_window[:, 1:], next_z.unsqueeze(1)], dim=1)
        history_z.append(next_z.squeeze().cpu().numpy())

    # --- RECONSTRUCT ---
    print("Reconstructing frames...")
    decoded_frames = []
    for tz in history_z:
        with torch.no_grad():
            # Get embedding from codebook
            # tz: (16, 16)
            tz_torch = torch.from_numpy(tz).to(DEVICE).long().view(1, -1)
            # Use vq_layer.embedding directly
            # quantized = vqvae.vq_layer._embedding(tz_torch).view(1, 16, 16, 64).permute(0, 3, 1, 2)
            # The above might be private, let's check vqvae definition
            # In our updated VQVAE: self.embedding (public)
            quantized = vqvae.vq_layer.embedding(tz_torch).view(1, 16, 16, 64).permute(0, 3, 1, 2)
            recon = vqvae.decoder(quantized)
            img = (recon.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            decoded_frames.append(img)

    # 4. Save GIF
    save_path = ARTIFACTS_PATH / "dream_real_data.gif"
    imageio.mimsave(save_path, decoded_frames, fps=10)
    print(f"✅ GIF saved to {save_path}")

if __name__ == "__main__":
    try:
        from train_vqvae import VQVAE
        from train_transformer_dynamics import WorldModelTransformer
        
        vqvae = VQVAE().to(DEVICE)
        vqvae.load_state_dict(torch.load(ARTIFACTS_PATH / "vqvae.pth", map_location=DEVICE))
        
        wm = WorldModelTransformer().to(DEVICE)
        wm.load_state_dict(torch.load(ARTIFACTS_PATH / "world_model_transformer.pth", map_location=DEVICE))
        
        generate_gif(vqvae, wm)
    except Exception as e:
        print(f"Test failed: {e}")