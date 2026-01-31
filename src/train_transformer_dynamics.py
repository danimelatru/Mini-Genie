import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

# --- IMPORTS ---
try:
    from train_vqvae import VQVAE
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from train_vqvae import VQVAE

# --- CONFIG ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WINDOW_SIZE = 4
VOCAB_SIZE = 512
NUM_ACTIONS = 8
ENTROPY_WEIGHT = 0.01
EMBED_DIM = 64
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tokens"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# --- DATASET ---
class TokenTransitionsDataset(Dataset):
    def __init__(self, token_dir, window_size=WINDOW_SIZE, limit=None):
        self.files = sorted(list(Path(token_dir).glob("*.npy")))
        if limit: self.files = self.files[:limit]
        self.window_size = window_size
        self.data = []
        
        print(f"Loading {len(self.files)} episodes for windowed training (window={window_size})...")
        
        for f in self.files:
            try:
                tokens = np.load(f) # (T, 16, 16)
                if len(tokens.shape) != 3: continue

                action_file = str(f).replace("tokens", "actions")
                if not os.path.exists(action_file):
                     action_file = str(f).replace("_tokens.npy", "_actions.npy")
                
                if not os.path.exists(action_file): continue
                actions = np.load(action_file)

                limit_len = min(len(tokens), len(actions))
                
                # Create windowed sequences
                for i in range(limit_len - window_size):
                    seq_tokens = tokens[i:i+window_size] # (W, 16, 16)
                    seq_actions = actions[i:i+window_size] # (W,)
                    target_token = tokens[i+window_size] # (16, 16)
                    
                    self.data.append((seq_tokens, target_token, seq_actions))
            except Exception as e: 
                print(f"Error loading {f}: {e}")
                continue
        
        print(f"Total Sequences: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_tokens, target_token, seq_actions = self.data[idx]
        return torch.LongTensor(seq_tokens), torch.LongTensor(target_token), torch.LongTensor(seq_actions)

# --- MODELS ---
class ActionRecognitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        # Process the last frame and the next frame to infer action
        self.conv_net = nn.Sequential(
            nn.Conv2d(EMBED_DIM * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS)
        )

    def forward(self, z_prev, z_next):
        # z_prev is the last frame of the context window
        emb_prev = self.embedding(z_prev).permute(0, 3, 1, 2)
        emb_next = self.embedding(z_next).permute(0, 3, 1, 2) 
        x = torch.cat([emb_prev, emb_next], dim=1)
        return self.head(self.conv_net(x))

class WorldModelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.action_emb = nn.Linear(NUM_ACTIONS, HIDDEN_DIM)
        # Position embedding for 16x16 grid + temporal encoding
        self.pos_emb = nn.Parameter(torch.randn(1, 16*16, HIDDEN_DIM))
        self.temp_emb = nn.Parameter(torch.randn(1, WINDOW_SIZE, HIDDEN_DIM))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=NUM_HEADS, batch_first=True, dim_feedforward=HIDDEN_DIM*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, z_seq, action_probs):
        # z_seq: (B, W, 16, 16)
        B, W, H, W_grid = z_seq.shape
        
        # Embed each frame and add spatial position
        # Reshape to (B*W, 16*16, HIDDEN_DIM)
        x = self.embedding(z_seq.view(B * W, -1)) + self.pos_emb
        
        # Mean pool or flatten the spatial dimensions? 
        # For dynamics, we want to keep spatial if possible, but the sequence is (B, W, H, W, D)
        # Let's flatten spatial and keep temporal
        x = x.view(B, W, H * W_grid, HIDDEN_DIM)
        
        # Add temporal embedding to each frame's representation
        x = x + self.temp_emb.unsqueeze(2)
        
        # Flatten temporal and spatial: (B, W*H*W_grid, HIDDEN_DIM)
        x = x.view(B, W * H * W_grid, HIDDEN_DIM)
        
        # Inject action (broadcasted to all tokens in the sequence)
        act_v = self.action_emb(action_probs).unsqueeze(1)
        x = x + act_v 
        
        out = self.transformer(x)
        
        # Map back to 16x16 grid for the NEXT frame (target is just one frame)
        # We take the representation corresponding to the LAST frame in the window
        last_frame_out = out.view(B, W, H * W_grid, HIDDEN_DIM)[:, -1, :, :]
        
        return self.head(last_frame_out).view(B, 16, 16, VOCAB_SIZE)

# --- MAIN ---
def main():
    print(f"Step 2: Training (Entropy Lambda={ENTROPY_WEIGHT})...")
    dataset = TokenTransitionsDataset(DATA_PATH, limit=5000)
    if len(dataset) == 0: return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    action_net = ActionRecognitionNet().to(DEVICE)
    world_model = WorldModelTransformer().to(DEVICE)
    optimizer = optim.Adam(list(action_net.parameters()) + list(world_model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        total_ent = 0
        
        for z_seq, z_next, real_act in dataloader:
            z_seq, z_next, real_act = z_seq.to(DEVICE), z_next.to(DEVICE), real_act.to(DEVICE)
            optimizer.zero_grad()
            
            # Predict action from the transition between last frame in window and the next frame
            z_last = z_seq[:, -1, :, :]
            action_logits = action_net(z_last, z_next)
            action_probs = torch.softmax(action_logits, dim=1)
            
            # Predict next frame from sequence + action
            pred_logits = world_model(z_seq, action_probs)
            
            loss_recon = criterion(pred_logits.view(-1, VOCAB_SIZE), z_next.view(-1))
            
            # Entropy calculation for latent actions
            log_probs = torch.log_softmax(action_logits, dim=1)
            entropy = -(action_probs * log_probs).sum(dim=1).mean()
            
            # Total Loss
            loss = loss_recon + (ENTROPY_WEIGHT * entropy)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_ent += entropy.item()
            
            # For accuracy, take the last action in the sequence or the real_act if it's the target action
            # The dataset provides seq_actions where real_act is the one that led to z_next
            target_act = real_act[:, -1]
            total_acc += (torch.argmax(action_probs, dim=1) == target_act).float().mean().item()
            
        avg_ent = total_ent/len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} (Ent: {avg_ent:.4f}) | Acc: {total_acc/len(dataloader)*100:.1f}%")

    torch.save(action_net.state_dict(), ARTIFACTS_PATH / "action_net_transformer.pth")
    torch.save(world_model.state_dict(), ARTIFACTS_PATH / "world_model_transformer.pth")
    
    print("Generating Viz...")
    action_net.eval()
    all_preds, all_real = [], []
    with torch.no_grad():
        for z_seq, z_next, real_act in dataloader:
            z_seq, z_next = z_seq.to(DEVICE), z_next.to(DEVICE)
            z_last = z_seq[:, -1, :, :]
            all_preds.extend(torch.argmax(action_net(z_last, z_next), dim=1).cpu().numpy())
            all_real.extend(real_act[:, -1].cpu().numpy())
            
    cm = confusion_matrix(all_real, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Ground Truth Action')
    plt.xlabel('Discovered Cluster')
    plt.title(f'Latent Action Discovery (Entropy w={ENTROPY_WEIGHT})')
    plt.savefig(ARTIFACTS_PATH / "transformer_confusion_matrix.png")
    
    try:
        import generate_dream_gif
        generate_dream_gif.generate_gif(VQVAE().to(DEVICE), world_model)
    except: pass

if __name__ == "__main__":
    main()