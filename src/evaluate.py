'''from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class SummaryDataset(Dataset):
    def __init__(self, documents, summaries):
        self.documents = documents
        self.summaries = summaries

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.summaries[idx]

# Load your preprocessed data
documents = ["...", "...", "..."]  # Replace with your preprocessed documents
summaries = ["...", "...", "..."]  # Replace with your preprocessed summaries

# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Create a PyTorch Dataset
dataset = SummaryDataset(documents, summaries)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - t