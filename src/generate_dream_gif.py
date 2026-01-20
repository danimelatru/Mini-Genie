import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import random
from pathlib import Path

# --- IMPORT MODELS ---
try:
    from train_vqvae import VQVAE
    from train_transformer_dynamics import WorldModelTransformer
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from train_vqvae import VQVAE
    from train_transformer_dynamics import WorldModelTransformer

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
EPISODES_PATH = PROJECT_ROOT / "data" / "episodes" # Path to real data
GIF_PATH = ARTIFACTS_PATH / "dream_real_data.gif"

TOKEN_VOCAB = 512
NUM_ACTIONS = 8

def load_models():
    print(f"Loading models from {ARTIFACTS_PATH}...")
    
    vqvae = VQVAE().to(DEVICE)
    try:
        vqvae.load_state_dict(torch.load(ARTIFACTS_PATH / "vqvae.pth", map_location=DEVICE))
    except Exception as e:
        print(f"Error loading VQVAE: {e}")
        sys.exit(1)
    vqvae.eval()

    wm = WorldModelTransformer().to(DEVICE)
    try:
        wm.load_state_dict(torch.load(ARTIFACTS_PATH / "world_model_transformer.pth", map_location=DEVICE))
    except Exception as e:
        print(f"Error loading World Model: {e}")
        sys.exit(1)
    wm.eval()
    
    return vqvae, wm

def get_real_initial_state(vqvae):
    """
    Loads a REAL frame from the dataset to seed the dream.
    Using artificial frames (gray background) confuses the VQ-VAE trained on black backgrounds.
    """
    files = sorted(list(EPISODES_PATH.glob("*.npz")))
    if not files:
        print("Error: No episode files found in data/episodes!")
        sys.exit(1)
        
    print(f"Searching {len(files)} episodes for a valid start frame...")
    
    # Try random files until we find a good frame (not empty)
    for _ in range(20):
        f = random.choice(files)
        try:
            with np.load(f) as data:
                frames = data["frames"] # (T, 64, 64, 3)
                
                # Pick a random frame from the middle of the episode
                if len(frames) > 10:
                    idx = random.randint(0, len(frames) - 10)
                    real_frame = frames[idx]
                    
                    # Check if frame is not completely black (has some activity)
                    if np.mean(real_frame) > 5: # Threshold for non-black
                        print(f"Found valid seed frame in {f.name} at index {idx}")
                        
                        # Convert to Tensor (1, 3, 64, 64) normalized 0-1
                        frame_tensor = torch.from_numpy(real_frame).float() / 255.0
                        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                        
                        # Encode
                        with torch.no_grad():
                            z = vqvae.encoder(frame_tensor)
                            out = vqvae.vq_layer(z)
                            indices = out[2] # get indices
                            
                            # Flatten if needed or reshape
                            if indices.shape[-1] == 1:
                                indices = indices.view(1, 16, 16)
                            return indices, real_frame
        except Exception as e:
            continue
            
    print("Warning: Could not find interesting frame, using random noise.")
    return None, None

def enhance_contrast(img):
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-5: return img
    return (img - img_min) / (img_max - img_min)

def generate_gif(vqvae, wm):
    print("Generating DREAM from REAL SEED...")
    frames = []
    
    # Get Real Seed
    curr_z, original_img_np = get_real_initial_state(vqvae)
    
    # Add original frame as the first frame of GIF (for reference)
    if original_img_np is not None:
        frames.append(original_img_np / 255.0) # Normalize to 0-1 for plot
    
    steps_per_action = 6 
    
    with torch.no_grad():
        emb_obj = vqvae.vq_layer._embedding
        emb_table = emb_obj.weight if isinstance(emb_obj, nn.Embedding) else emb_obj

        # Test the most distinct actions from your matrix:
        # 0 (Movement Cluster) and 2 (Interaction Cluster)
        actions_to_test = [0, 2, 4] 
        
        for action_idx in actions_to_test:
            print(f"Dreaming Action {action_idx}...")
            
            action_probs = torch.zeros(1, NUM_ACTIONS).to(DEVICE)
            action_probs[0, action_idx] = 1.0
            
            for _ in range(steps_per_action):
                logits = wm(curr_z, action_probs)
                
                # --- GREEDY DECODING (DETERMINISTIC) ---
                # We select the most probable token to avoid noise.
                # Since the model is not fully converged, sampling introduces gray artifacts.
                next_indices = torch.argmax(logits, dim=2)
                
                # Decode
                z_q = torch.nn.functional.embedding(next_indices, emb_table)
                z_q = z_q.view(1, 16, 16, 64).permute(0, 3, 1, 2)
                recon = vqvae.decoder(z_q)
                
                img = recon.squeeze().permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                
                # Slight contrast enhance
                # img = enhance_contrast(img) 
                
                frames.append(img)
                curr_z = next_indices.view(1, 16, 16)

    print(f"Saving GIF to {GIF_PATH}...")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    im = ax.imshow(frames[0], animated=True)
    
    def update(i):
        im.set_array(frames[i])
        if i == 0:
            ax.set_title("SEED FRAME (Real)", color='green')
        else:
            # Adjust index for title because frame 0 is seed
            act_idx = ((i-1) // steps_per_action)
            if act_idx < len(actions_to_test):
                ax.set_title(f"Dreaming Action: {actions_to_test[act_idx]}", color='blue')
        return im,
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=250, blit=True)
    ani.save(str(GIF_PATH), writer='pillow')
    print("Done!")

if __name__ == "__main__":
    models = load_models()
    generate_gif(*models)