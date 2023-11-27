import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from vocab import Vocabulary
from datasets import SentimentalSummarizationDataset
from models_ssum_attention import *

data = pd.read_csv("data/amazon_food_review.tsv", sep="\t")
data = data.dropna()

max_text_len = 180
max_summary_len = 10

cleaned_text =np.array(data['Text'])
cleaned_summary=np.array(data['Summary'])
cleaned_score = np.array(data['Score'])

short_text = []
short_summary = []
short_score = []

for i in range(len(cleaned_text)):
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])
        short_score.append(cleaned_score[i])
        
df=pd.DataFrame({'text':short_text,'summary':short_summary, 'score':short_score})

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

train_data, temp_data = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0, shuffle=True)

vocab = Vocabulary(freq_threshold=6)
all_texts = [text for text in df['text']] + [summary for summary in df['summary']]
vocab.build_vocabulary(all_texts)

train_dataset = SentimentalSummarizationDataset(train_data['text'].values, train_data['summary'].values, train_data['score'].values, vocab)
valid_dataset = SentimentalSummarizationDataset(valid_data['text'].values, valid_data['summary'].values, valid_data['score'].values, vocab)
test_dataset = SentimentalSummarizationDataset(test_data['text'].values, test_data['summary'].values, test_data['score'].values, vocab)

def collate_fn(batch):
    originals, summaries, labels = zip(*batch)
    originals_padded = pad_sequence(originals, padding_value=vocab.stoi["<PAD>"])
    summaries_padded = pad_sequence(summaries, padding_value=vocab.stoi["<PAD>"])
    labels = torch.stack(labels)

    return originals_padded, summaries_padded, labels

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

def train(model_summary, model_sentiment, iterator, optimizer_summary, optimizer_sentiment, criterion_summary, criterion_sentiment, clip):
    model_summary.train()
    model_sentiment.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        src, trg, label = batch
        src, trg, label = src.to(device), trg.to(device), label.to(device)
        label = label.squeeze()
        optimizer_sentiment.zero_grad()
        output_sentiment = model_sentiment(src)
        loss_sentiment = criterion_sentiment(output_sentiment, label)
        loss_sentiment.backward()
        torch.nn.utils.clip_grad_norm_(model_sentiment.parameters(), clip)
        optimizer_sentiment.step()

        optimizer_summary.zero_grad()
        sentiment_hidden = model_sentiment.hidden(src)
        output = model_summary(src, trg[:-1], sentiment_hidden)
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss_summary = criterion_summary(output, trg)
        loss_summary.backward()

        torch.nn.utils.clip_grad_norm_(model_summary.parameters(), clip)

        optimizer_summary.step()

        epoch_loss += loss_summary.item()

    return epoch_loss / len(iterator)

def evaluate(model_summary, model_sentiment, iterator, criterion_summary):
    model_summary.eval()
    model_sentiment.eval()  
    epoch_loss = 0

    with torch.no_grad():  
        for i, batch in enumerate(tqdm(iterator)):
            src, trg, label = batch
            src, trg, label = src.to(device), trg.to(device), label.to(device)
            label = label.squeeze()
            sentiment_hidden = model_sentiment.hidden(src)
            output = model_summary(src, trg[:-1], sentiment_hidden, 0) 

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion_summary(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


device = torch.device("cuda")
input_dim = len(vocab)
output_dim = len(vocab)
encoder = EncoderLSTM(input_dim, 256, 512, 2, 0.5)
decoder = DecoderLSTM(output_dim, 256, 512, 2, 0.5)
model_summary = Seq2Seq(encoder, decoder, device).to(device)

optimizer_summary = optim.Adam(model_summary.parameters(),)
pad_idx = vocab.stoi["<PAD>"]
criterion_summary = nn.CrossEntropyLoss(ignore_index=pad_idx)

model_sentiment = SentimentLSTM(input_dim, 256, 512, 5, 2, 0.5).to(device)
optimizer_sentiment = optim.Adam(model_sentiment.parameters())
criterion_sentiment = nn.CrossEntropyLoss()
num_epochs = 10
clip = 1

for epoch in range(num_epochs):
    train_loss = train(model_summary=model_summary, model_sentiment=model_sentiment, iterator=train_loader,\
                    optimizer_summary=optimizer_summary, optimizer_sentiment=optimizer_sentiment, \
                    criterion_summary=criterion_summary, criterion_sentiment=criterion_sentiment, clip=clip)
    valid_loss = evaluate(model_summary=model_summary, model_sentiment=model_sentiment, iterator=valid_loader, \
                    criterion_summary=criterion_summary)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

test_loss = evaluate(model_summary=model_summary, model_sentiment=model_sentiment, iterator=test_loader, \
                    criterion_summary=criterion_summary)
print(f'Test Loss: {test_loss:.3f}')