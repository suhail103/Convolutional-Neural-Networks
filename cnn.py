import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import json
import unicodedata
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt


class Word2Vec(nn.Module):
    
    def __init__(self, dictionary_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()
        self.embedding_layer = nn.Embedding(dictionary_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, dictionary_size)
        
        for layer in [self.embedding_layer, self.output_layer]:
            if not isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.bias, mean=0.0, std=0.05).clamp(-0.1, 0.1)
            nn.init.normal_(layer.weight, mean=0.0, std=0.05).clamp(-0.1, 0.1)
    
    def forward(self, x):
        z = self.embedding_layer(x)
        logits = self.output_layer(z)
        return logits

def train_word2vec_encoder(files, embedding_dims=128, window_size=2, epochs=5):
    
    print("\n[+] Starting Word2Vec encoder training...")
    
    text = []
    for file_path in files:
        text.extend(get_book_text(file_path))

    unique_words = sorted(list(set(text)))
    word_to_index = {word: index for index, word in enumerate(unique_words)}
    
    dataset = []
    for i in range(window_size, len(text) - window_size):
        for j in range(1, window_size + 1):
            dataset.append([word_to_index[text[i]], word_to_index[text[i - j]]])
            dataset.append([word_to_index[text[i]], word_to_index[text[i + j]]])
    
    dictionary_size = len(unique_words)
    model = Word2Vec(dictionary_size, embedding_dims)
    
    dataset_tensor = T.tensor(dataset, dtype=T.long)
    dataloader = DataLoader(TensorDataset(dataset_tensor[:, 0], dataset_tensor[:, 1]), batch_size=256, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = T.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (X, y_true) in enumerate(dataloader):
            logits = model(X)
            loss = criterion(logits, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Encoder Training Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    print("[+] Word2Vec encoder training complete.")
    return model.embedding_layer, word_to_index


def remove_punctuations(text):
    
    return "".join(filter(lambda x: not unicodedata.category(x).startswith('P'), text.replace('-', ' ')))

def get_book_text(file_path):
    
    text = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            no_punctuation_line = remove_punctuations(line)
            for word in no_punctuation_line.split():
                if len(word) < 2: continue
                text.append(word.lower())
    return text


def create_sequences(files, word_to_index, seq_length=200):
    
    all_sequences = []
    all_labels = []

    for label, file_path in enumerate(files):
        print(f"Processing {file_path} with label {label}...")
        text = get_book_text(file_path)
        
        for i in range(0, len(text) - seq_length, seq_length):
            sequence_words = text[i:i + seq_length]
            indexed_sequence = [word_to_index.get(word, 0) for word in sequence_words]
            all_sequences.append(indexed_sequence)
            all_labels.append(label)

    return T.tensor(all_sequences, dtype=T.long), T.tensor(all_labels, dtype=T.long)


class BaseCNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx, pretrained_embeddings):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(T.cat(pooled, dim = 1))
        return self.fc(cat)

class DeepCNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx, pretrained_embeddings):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False
        
        self.convs1 = nn.ModuleList([
                                    nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
                                    for fs in filter_sizes
                                    ])
        self.convs2 = nn.ModuleList([
                                    nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved1 = [F.relu(conv(embedded)) for conv in self.convs1]
        conved2 = [F.relu(conv(out)) for conv, out in zip(self.convs2, conved1)]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved2]
        cat = self.dropout(T.cat(pooled, dim = 1))
        return self.fc(cat)


def train_classifier(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for text, labels in iterator:
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        
        preds_class = T.argmax(F.softmax(predictions, dim=1), dim=1)
        correct = (preds_class == labels).float()
        acc = correct.sum() / len(correct)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_classifier(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with T.no_grad():
        for text, labels in iterator:
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            preds_class = T.argmax(F.softmax(predictions, dim=1), dim=1)
            correct = (preds_class == labels).float()
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

if __name__ == "__main__":
    print("[+] Starting Lab 2: CNN Classifier")
    
    files = [f"./Harry_Potter_Books/HP{i}.txt" for i in range(1, 8)]

    trained_embedding_layer, word_to_index = train_word2vec_encoder(files, epochs=5)
    pretrained_embeddings = trained_embedding_layer.weight.data
    
    VOCAB_SIZE = len(word_to_index)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    sequences, labels = create_sequences(files, word_to_index, seq_length=200)

    print(f"\n[+] Dataset created successfully.")
    print(f"    Total sequences (pages): {len(sequences)}")
    print(f"    Total labels: {len(labels)}")

    dataset = TensorDataset(sequences, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"\n[+] Splitting data into training and testing sets.")
    print(f"    Training samples: {len(train_dataset)}")
    print(f"    Testing samples: {len(test_dataset)}")

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print(f"\n[+] DataLoaders created with batch size {BATCH_SIZE}.")

    EMBEDDING_DIM = 128
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 7
    DROPOUT = 0.5
    PAD_IDX = 0
    N_EPOCHS = 15
    
    print("\n[+] Initializing the Base CNN model...")
    base_model = BaseCNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX, pretrained_embeddings)
    optimizer = T.optim.Adam(base_model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    base_train_loss_history = []
    
    print(f"\n[+] Starting training for Base Model ({N_EPOCHS} epochs)...")
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_classifier(base_model, train_loader, optimizer, criterion)
        base_train_loss_history.append(train_loss)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    test_loss_base, test_acc_base = evaluate_classifier(base_model, test_loader, criterion)
    
    print("\n[+] Initializing the Deep CNN model...")
    deep_model = DeepCNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX, pretrained_embeddings)
    optimizer = T.optim.Adam(deep_model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    deep_train_loss_history = []

    print(f"\n[+] Starting training for Deep Model ({N_EPOCHS} epochs)...")
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_classifier(deep_model, train_loader, optimizer, criterion)
        deep_train_loss_history.append(train_loss)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    test_loss_deep, test_acc_deep = evaluate_classifier(deep_model, test_loader, criterion)

    print(f'\n[+] --- FINAL RESULTS ---')
    print(f'Base Model (1 Layer) Test Accuracy: {test_acc_base*100:.2f}%')
    print(f'Deep Model (2 Layers) Test Accuracy: {test_acc_deep*100:.2f}%')

    print("\n[+] Generating and saving performance graphs...")
    
    epochs_range = range(1, N_EPOCHS + 1)
    
    plt.figure()
    plt.plot(epochs_range, base_train_loss_history, 'b-o')
    plt.title('Base Model Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('base_model_loss.png')
    print("[+] Graph 'base_model_loss.png' saved successfully.")
    plt.close() 

    plt.figure()
    plt.plot(epochs_range, deep_train_loss_history, 'r-o')
    plt.title('Deeper Model Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('deeper_model_loss.png')
    print("[+] Graph 'deeper_model_loss.png' saved successfully.")
    plt.close()
