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

# Load your preprocessed 