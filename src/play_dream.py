import torch
import torch.nn as nn
import numpy as np
import cv2
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

# --- SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
EPISODES_PATH = PROJECT_ROOT / "data" / "episodes" # Para coger una semilla real

TOKEN_VOCAB = 512
NUM_ACTIONS = 8

def load_models():
    print(f"Loading models from {ARTIFACTS_PATH}...")
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(ARTIFACTS_PATH / "vqvae.pth", map_location=DEVICE))
    vqvae.eval()

    wm = WorldModelTransformer().to(DEVICE)
    wm.load_state_dict(torch.load(ARTIFACTS_PATH / "world_model_transformer.pth", map_location=DEVICE))
    wm.eval()
    
    return vqvae, wm

def get_real_initial_state(vqvae):
    """Loads a random real frame to start the dream correctly."""
    files = sorted(list(EPISODES_PATH.glob("*.npz")))
    if not files:
        print("Error: No data found. Using dummy black frame.")
        return torch.zeros(1, 16, 16).long().to(DEVICE)
    
    # Try to find a non-empty frame
    for _ in range(10):
        f = random.choice(files)
        try:
            with np.load(f) as data:
                frames = data["frames"]
                if len(frames) > 10:
                    idx = random.randint(0, len(frames)-1)
                    real_frame = frames[idx]
                    if np.mean(real_frame) > 5: # Not black
                        frame_tensor = torch.from_numpy(real_frame).float() / 255.0
                        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            indices = vqvae.encoder(frame_tensor)
                            indices = vqvae.vq_layer(indices)[2]
                            if indices.shape[-1] == 1: indices = indices.view(1, 16, 16)
                            return indices
        except: continue
    return torch.zeros(1, 16, 16).long().to(DEVICE)

def main():
    print("Loading models...")
    try:
        vqvae, wm = load_models()
    except FileNotFoundError:
        print("Models not found! Train the model first.")
        return

    print("Initializing Dream with Real Data...")
    current_tokens = get_real_initial_state(vqvae)
    
    # Get embedding table safely
    emb_obj = vqvae.vq_layer._embedding
    emb_table = emb_obj.weight if isinstance(emb_obj, nn.Embedding) else emb_obj
    
    print("\n--- CONTROLS ---")
    print("KEYS 0-7: Change Latent Action")
    print("KEY  'r': Reset to new random real frame")
    print("KEY  'q': Quit")
    print("----------------")
    
    current_action_idx = 0
    
    while True:
        # 1. Prepare Action
        action_probs = torch.zeros(1, NUM_ACTIONS).to(DEVICE)
        action_probs[0, current_action_idx] = 1.0
        
        # 2. Dream Next Step
        with torch.no_grad():
            logits = wm(current_tokens, action_probs)
            
            # GREEDY DECODING (Crucial for stability)
            next_token_indices = torch.argmax(logits, dim=2) 
            
            # Decode
            z_q = torch.nn.functional.embedding(next_token_indices, emb_table)
            z_q = z_q.view(1, 16, 16, 64).permute(0, 3, 1, 2)
            decoded_frame = vqvae.decoder(z_q)
            
            # Update state
            current_tokens = next_token_indices.view(1, 16, 16)

        # 3. Render
        img = decoded_frame.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # GUI Info
        cv2.putText(img, f"Action: {current_action_idx}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Mini-Genie Interactive Dream", img)
        
        # 4. Input Handling
        key = cv2.waitKey(50) # 20 FPS
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting world...")
            current_tokens = get_real_initial_state(vqvae)
        elif key >= ord('0') and key <= ord('7'):
            current_action_idx = key - ord('0')
            print(f"Action switched to: {current_action_idx}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()