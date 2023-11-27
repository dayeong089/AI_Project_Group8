import torch
from torch.utils.data import Dataset

class TextSummarizationDataset(Dataset):
    def __init__(self, original_texts, summaries, vocab):
        self.original_texts = original_texts
        self.summaries = summaries
        self.vocab = vocab

    def __len__(self):
        return len(self.original_texts)

    def __getitem__(self, index):
        original_text = self.original_texts[index]
        summary = self.summaries[index]

        numericalized_original = [self.vocab.stoi["<SOS>"]]
        numericalized_original += self.vocab.numericalize(original_text)
        numericalized_original.append(self.vocab.stoi["<EOS>"])

        numericalized_summary = [self.vocab.stoi["<SOS>"]]
        numericalized_summary += self.vocab.numericalize(summary)
        numericalized_summary.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_original), torch.tensor(numericalized_summary)

class ReviewDataset(Dataset):
    def __init__(self, original_texts, labels, vocab):
        self.original_texts = original_texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.original_texts)
    
    def __getitem__(self, index):
        original_text = self.original_texts[index]
        label = self.labels[index]

        numericalized_original = [self.vocab.stoi["<SOS>"]]
        numericalized_original += self.vocab.numericalize(original_text)
        numericalized_original.append(self.vocab.stoi["<EOS>"])
        return torch.tensor(numericalized_original), torch.tensor([label-1], dtype=torch.long)

class SentimentalSummarizationDataset(Dataset):
    def __init__(self, original_texts, summaries, labels, vocab):
        self.original_texts = original_texts
        self.summaries = summaries
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.original_texts)

    def __getitem__(self, index):
        original_text = self.original_texts[index]
        summary = self.summaries[index]
        label = self.labels[index]

        numericalized_original = [self.vocab.stoi["<SOS>"]]
        numericalized_original += self.vocab.numericalize(original_text)
        numericalized_original.append(self.vocab.stoi["<EOS>"])

        numericalized_summary = [self.vocab.stoi["<SOS>"]]
        numericalized_summary += self.vocab.numericalize(summary)
        numericalized_summary.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_original), torch.tensor(numericalized_summary), torch.tensor([label-1], dtype=torch.long)