import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_dataset import load_and_preprocess_data, build_vocab, create_dataloaders
from models import AttentionClassifier, RNNClassifier, ManualAttentionClassifier

def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=5, lr=1e-3, device='cuda'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n--- Training {model.__class__.__name__} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Test Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds)
    r = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"Results for {model.__class__.__name__}:")
    print(f"Accuracy: {acc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    return acc, f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_preprocess_data()
    exp_results = []
    vocab = build_vocab(train_texts, max_vocab_size=30000)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, 
        vocab, max_seq_len=200, batch_size=128
    )
    
    vocab_size = len(vocab)
    
    # 2. Base Models
    attention_model = AttentionClassifier(vocab_size=vocab_size)
    rnn_model = RNNClassifier(vocab_size=vocab_size)
    
    # 3. Additional Experiment Models
    manual_attn_model = ManualAttentionClassifier(vocab_size=vocab_size)
    no_pe_attn_model = AttentionClassifier(vocab_size=vocab_size, use_pe=False)
    
    # Run evaluations (You can comment/uncomment to run specific tests)
    # train_and_evaluate(attention_model, train_loader, val_loader, test_loader, device=device)
    # train_and_evaluate(rnn_model, train_loader, val_loader, test_loader, device=device)
    '''print("\n=== Experiment 1: Number of Attention Heads ===")
    for h in [2, 4, 8]:
        print(f"\nTesting num_heads = {h}")
        model = AttentionClassifier(vocab_size=len(vocab), num_heads=h, embed_dim=128)
        acc, f1 = train_and_evaluate(model, train_loader, None, test_loader, device=device)
        exp_results.append({"Group": "num_heads", "Value": h, "Acc": acc, "F1": f1})
    print("\n=== Experiment 2: Learning Rate ===")
    for lr in [1e-4, 1e-3, 5e-3]:
        print(f"\nTesting lr = {lr}")
        model = AttentionClassifier(vocab_size=len(vocab))
        acc, f1 = train_and_evaluate(model, train_loader, None, test_loader, lr=lr, device=device)
        exp_results.append({"Group": "learning_rate", "Value": lr, "Acc": acc, "F1": f1}) 
    print("\n=== Experiment 3: Max Vocab Size ===")
    for v_size in [10000, 20000, 30000]:
        print(f"\nTesting vocab_size = {v_size}")
        temp_vocab = build_vocab(train_texts, max_vocab_size=v_size)
        t_loader, _, te_loader = create_dataloaders(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, temp_vocab,max_seq_len=200)'''
    # 在 train.py 的实验部分新增
    print("\n--- Training ManualAttentionClassifier with RoPE ---")
    rope_model = ManualAttentionClassifier(
        vocab_size=len(vocab), 
        pe_type='rope',  # 指定使用 RoPE
    ).to(device)
    acc_rope, f1_rope = train_and_evaluate(rope_model, train_loader, val_loader, test_loader, device=device)
    # 附加实验1: Manual MHA
    # train_and_evaluate(manual_attn_model, train_loader, val_loader, test_loader, device=device)
    # 附加实验3: 无位置编码
    # train_and_evaluate(no_pe_attn_model, train_loader, val_loader, test_loader, device=device)
