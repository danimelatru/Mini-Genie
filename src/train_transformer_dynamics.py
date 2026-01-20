import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# --- CONFIGURATION ---
BATCH_SIZE = 64              # Reduced batch size for Transformer memory
LEARNING_RATE = 3e-4         # Lower LR for Transformer stability
EPOCHS = 100                  # Increased epochs to give time for convergence
NUM_LATENT_ACTIONS = 8       # Number of discrete actions to discover
TOKEN_VOCAB = 512            # Must match VQ-VAE codebook size
SEQ_LEN = 256                # 16x16 tokens flattened
EMBED_DIM = 128              # Transformer internal dimension
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATH SETUP ---
current_path = Path(__file__).parent.resolve()
root_path = current_path.parent
sys.path.append(str(root_path))
DATA_PATH = root_path / "data" / "tokens"
ARTIFACTS_PATH = root_path / "data" / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# --- 1. DATASET ---
class TokenTransitionsDataset(Dataset):
    def __init__(self, data_dir, limit=10000, min_diff_tokens=1):
        self.files = sorted(list(Path(data_dir).glob("*.npz")))
        if len(self.files) > limit:
            self.files = self.files[:limit]
        
        print(f"Loading {len(self.files)} episodes...")
        
        self.transitions = []
        self.real_actions = [] 
        
        total_transitions = 0
        kept_transitions = 0
        
        for f in self.files:
            with np.load(f) as data:
                tokens = data["tokens"] # (T, 16, 16)
                actions = data["actions"] # (T, 2)
                
                # --- FILTRO DE MOVIMIENTO ---
                # Compute how many tokens change between t and t+1
                # tokens[:-1] is the actual state, tokens[1:] is the next one
                diff_map = (tokens[:-1] != tokens[1:])
                # Sum all changes per frame (axis 1 and 2)
                changes_per_frame = np.sum(diff_map, axis=(1, 2))
                
                # Only keep index with at least 'min_diff_tokens' changes
                # Deletes frames where agent stayed still or the VQVAE didn't percieve movement
                interesting_indices = np.where(changes_per_frame >= min_diff_tokens)[0]
                
                if len(interesting_indices) == 0:
                    continue
                    
                # Select interesting frames
                current_steps = tokens[interesting_indices]
                next_steps = tokens[interesting_indices + 1]
                agent_0_actions = actions[interesting_indices, 0] 
                
                self.transitions.append(np.stack([current_steps, next_steps], axis=1))
                self.real_actions.append(agent_0_actions)
                
                total_transitions += (len(tokens) - 1)
                kept_transitions += len(interesting_indices)
                
        self.transitions = np.concatenate(self.transitions, axis=0)
        self.real_actions = np.concatenate(self.real_actions, axis=0)
        
        print(f"--- DATASET CURATION REPORT ---")
        print(f"Original Transitions: {total_transitions}")
        print(f"Kept (Moving) Transitions: {kept_transitions}")
        print(f"Discarded (Static): {total_transitions - kept_transitions} ({100*(1 - kept_transitions/total_transitions):.1f}%)")
        print(f"This forces the model to learn DYNAMICS, not background copying.")
        print(f"-------------------------------")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        trans = torch.from_numpy(self.transitions[idx]).long()
        real_act = torch.tensor(self.real_actions[idx]).long()
        return trans[0], trans[1], real_act

# --- 2. MODELS ---

class ActionRecognitionNet(nn.Module):
    """
    Inverse Model: Infers action a_t from state z_t and z_{t+1}.
    Uses CNNs because movement is a spatial phenomenon.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOKEN_VOCAB, 32)
        
        # Input: 2 frames stacked (channels)
        # 16x16 input -> CNN -> Global Average Pooling
        self.conv_net = nn.Sequential(
            nn.Conv2d(32 * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2), # 4x4
            nn.ReLU()
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_LATENT_ACTIONS)
        )

    def forward(self, z_t, z_next):
        # z: (B, 16, 16)
        emb_t = self.embedding(z_t).permute(0, 3, 1, 2)       # (B, 32, 16, 16)
        emb_next = self.embedding(z_next).permute(0, 3, 1, 2) # (B, 32, 16, 16)
        
        x = torch.cat([emb_t, emb_next], dim=1) # (B, 64, 16, 16)
        feat = self.conv_net(x)
        logits = self.head(feat)
        return logits


class WorldModelTransformer(nn.Module):
    """
    Forward Model: Predicts z_{t+1} given z_t and action a_t.
    Architecture: GPT-style Decoder (Causal/Parallel).
    """
    def __init__(self):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(TOKEN_VOCAB, EMBED_DIM)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN, EMBED_DIM))
        self.action_embedding = nn.Embedding(NUM_LATENT_ACTIONS, EMBED_DIM)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=NUM_HEADS, 
            dim_feedforward=EMBED_DIM*4, 
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        # Output Head
        self.head = nn.Linear(EMBED_DIM, TOKEN_VOCAB)

    def forward(self, z_t, action_probs):
        # z_t: (B, 16, 16) -> Flatten to (B, 256)
        b, h, w = z_t.shape
        z_flat = z_t.view(b, -1)
        
        # 1. Embed Tokens
        x = self.token_embedding(z_flat) # (B, 256, D)
        
        # 2. Add Positional Embeddings
        x = x + self.pos_embedding
        
        # 3. Inject Action Information
        # Compute the expected action embedding using the probabilities from the Inverse Model
        # action_probs: (B, NUM_ACTIONS)
        # action_emb_matrix: (NUM_ACTIONS, D)
        act_emb = torch.matmul(action_probs, self.action_embedding.weight) # (B, D)
        act_emb = act_emb.unsqueeze(1) # (B, 1, D)
        
        # Broadcast action to all tokens (Conditioning)
        # Alternatively, we could prepend it as a token, but adding it preserves length.
        x = x + act_emb 
        
        # 4. Transformer Pass
        # We don't use a causal mask here because we are predicting t+1 from t (all-to-all attention is fine)
        feat = self.transformer(x)
        
        # 5. Predict Next Tokens
        logits = self.head(feat) # (B, 256, Vocab)
        
        return logits

# --- 3. TRAINING LOOP ---
def main():
    print(f"Running TRANSFORMER Dynamics training on {DEVICE}")
    
    # Increased limit to 10k to provide enough variety for the Transformer
    dataset = TokenTransitionsDataset(DATA_PATH, limit=10000) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    action_net = ActionRecognitionNet().to(DEVICE)
    world_model = WorldModelTransformer().to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(action_net.parameters()) + list(world_model.parameters()), 
        lr=LEARNING_RATE
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Training (NO ENTROPY REGULARIZATION)...")
    action_net.train()
    world_model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_recon = 0
        correct_actions = 0
        total_samples = 0
        
        for curr_z, next_z, real_act in dataloader:
            curr_z, next_z, real_act = curr_z.to(DEVICE), next_z.to(DEVICE), real_act.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. Infer Action (Inverse Dynamics)
            action_logits = action_net(curr_z, next_z)
            action_probs = F.softmax(action_logits, dim=1) # Differentiable bottleneck
            
            # 2. Predict Future (Forward Dynamics)
            # Pass gradients through action_probs to train the ActionNet
            pred_next_logits = world_model(curr_z, action_probs)
            
            # 3. Calculate Losses
            # Flatten for CrossEntropy: (B*256, Vocab) vs (B*256)
            recon_loss = criterion(
                pred_next_logits.view(-1, TOKEN_VOCAB), 
                next_z.view(-1)
            )
            
            # --- NO ENTROPY REGULARIZATION ---
            # Removed the entropy term to force the model to minimize reconstruction error by finding meaningful actions, rather than maximizing randomness.
            loss = recon_loss 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            
            # Calculate Accuracy (for monitoring purposes only)
            # This checks if the learned latent actions map to real keyboard inputs
            pred_act = torch.argmax(action_logits, dim=1)
            correct_actions += (pred_act == real_act).sum().item()
            total_samples += real_act.size(0)
            
        acc = correct_actions / total_samples
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Acc Real: {acc:.2%} (Chance: {100/NUM_LATENT_ACTIONS:.1f}%)")

    # --- SAVE ---
    torch.save(action_net.state_dict(), ARTIFACTS_PATH / "action_net_transformer.pth")
    torch.save(world_model.state_dict(), ARTIFACTS_PATH / "world_model_transformer.pth")
    print("Models saved.")

    # --- EVALUATION ---
    print("Generating Confusion Matrix...")
    action_net.eval()
    all_pred = []
    all_real = []
    
    with torch.no_grad():
        for curr_z, next_z, real_act in dataloader:
            curr_z, next_z = curr_z.to(DEVICE), next_z.to(DEVICE)
            logits = action_net(curr_z, next_z)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_real.extend(real_act.numpy())
            
    cm = confusion_matrix(all_real, all_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Latent Action Discovery (NO REGULARIZATION)')
    plt.xlabel('Discovered Cluster')
    plt.ylabel('Ground Truth Action')
    plt.savefig(ARTIFACTS_PATH / "transformer_confusion_matrix.png")

if __name__ == "__main__":
    main()