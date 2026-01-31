import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import sys
import os

# --- IMPORTS ---
# Attempt to import models. If it fails, we handle it or rely on local definitions.
try:
    from train_vqvae import VQVAE
    from train_transformer_dynamics import WorldModelTransformer, ActionRecognitionNet
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    try:
        from train_vqvae import VQVAE
        from train_transformer_dynamics import WorldModelTransformer, ActionRecognitionNet
    except ImportError:
        print("Error: Could not import models. Make sure train_vqvae.py and train_transformer_dynamics.py are in src/")
        sys.exit(1)

# --- CONFIG ---
BATCH_SIZE = 64
WINDOW_SIZE = 4
VOCAB_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tokens"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
ACTIONS_PATH = PROJECT_ROOT / "data" / "actions"

# Redefine Dataset here to make this script standalone and robust
class TokenTransitionsDataset(Dataset):
    def __init__(self, token_dir, limit=None):
        self.files = sorted(list(Path(token_dir).glob("*.npy")))
        if limit: self.files = self.files[:limit]
        self.data = []
        for f in self.files:
            try:
                tokens = np.load(f)
                if len(tokens.shape) != 3: continue
                
                # Logic to find the corresponding action file
                action_file = str(f).replace("tokens", "actions")
                if not os.path.exists(action_file): 
                    action_file = str(f).replace("_tokens.npy", "_actions.npy")
                if not os.path.exists(action_file): continue
                
                actions = np.load(action_file)
                
                # Clamp length
                limit_len = min(len(tokens) - 1, len(actions))
                for i in range(limit_len):
                    curr = tokens[i]
                    nxt = tokens[i+1]
                    if np.array_equal(curr, nxt): continue
                    self.data.append((curr, nxt, actions[i]))
            except: continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        curr, nxt, act = self.data[idx]
        return torch.LongTensor(curr), torch.LongTensor(nxt), torch.LongTensor([act])

def main():
    print("🎨 Step 4: Generating final visualizations (t-SNE & GIF)...")
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset = TokenTransitionsDataset(DATA_PATH, limit=1000) # We only need a sample
    if len(dataset) == 0:
        print("No data found.")
        return
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Load Action Recognition Model
    print("Loading Action Recognition Model...")
    action_net = ActionRecognitionNet().to(DEVICE)
    try:
        action_net.load_state_dict(torch.load(ARTIFACTS_PATH / "action_net_transformer.pth", map_location=DEVICE))
    except FileNotFoundError:
        print("Model not found. Skipping t-SNE.")
        dataset = TokenTransitionsDataset(DATA_PATH, window_size=WINDOW_SIZE, limit=5000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Extracting latents for {len(dataset)} sequences...")
    
    all_latents = []
    all_actions = []
    
    with torch.no_grad():
            real_labels.extend(real_act.view(-1).cpu().numpy())
            pred_labels.extend(preds.view(-1).cpu().numpy())
            
            if len(real_labels) > 2000: break
            
    latents = np.concatenate(latents, axis=0)
    
    # 4. Run t-SNE
    print("Running t-SNE (Calculating 2D map)...")
    # Perplexity must be less than the number of samples
    perp = min(30, len(latents) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_embedded = tsne.fit_transform(latents)
    
    # 5. Plot
    print(f"Plotting to {ARTIFACTS_PATH}...")
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "Cluster": pred_labels,
        "Real Action": real_labels 
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x="x", y="y", 
        hue="Cluster", 
        palette="tab10",
        s=60, alpha=0.7
    )
    plt.title("Mini-Genie Brain Map: Discovered Action Clusters")
    save_path = ARTIFACTS_PATH / "action_latent_space_tsne.png"
    plt.savefig(save_path)
    print(f"t-SNE saved to {save_path}")
    
    # 6. Generate GIF (Dreaming)
    try:
        import generate_dream_gif
        print("Generating GIF...")
        
        # Load World Model
        world_model = WorldModelTransformer().to(DEVICE)
        world_model.load_state_dict(torch.load(ARTIFACTS_PATH / "world_model_transformer.pth", map_location=DEVICE))
        
        # Load VQVAE (Eyes)
        vqvae = VQVAE().to(DEVICE)
        vqvae.load_state_dict(torch.load(ARTIFACTS_PATH / "vqvae.pth", map_location=DEVICE))
        
        generate_dream_gif.generate_gif(vqvae, world_model)
    except Exception as e:
        print(f"Could not generate GIF: {e}")

if __name__ == "__main__":
    main()