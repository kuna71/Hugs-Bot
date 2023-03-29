import torch
from torch import nn
from transformers import AutoModel, BertTokenizerFast
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
class BERT_architecture(nn.Module):
  
    def __init__(self, bert):
        
      super(BERT_architecture, self).__init__()
  
      self.bert = bert 
        
      # dropout layer
      self.dropout = nn.Dropout(0.2)
        
      # relu activation function
      self.relu =  nn.ReLU()
  
      # dense layer 1
      self.fc1 = nn.Linear(768,512)
        
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)
  
      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)
  
    #define the forward pass
    def forward(self, sent_id, mask):
  
      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
      x = self.fc1(cls_hs)
  
      x = self.relu(x)
  
      x = self.dropout(x)
  
      # output layer
      x = self.fc2(x)
        
      # apply softmax activation
      x = self.softmax(x)
  
      return x
def IsDepressed(text, pad_len):
    device = torch.device("cpu")
    model = BERT_architecture(bert)
    model.load_state_dict(torch.load("model.pt"))
    tokens = tokenizer.encode_plus(
        text,
        padding=True,
        max_length = pad_len,
        truncation = True)
    tokens_seq = torch.tensor(tokens['input_ids']).unsqueeze(0)
    tokens_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)
    preds = model.forward(tokens_seq.to(device), tokens_mask.to(device))
    preds = preds.detach().cpu().numpy()
    print(preds)
    if(preds[0][1]>-0.6):
        return True
    else:
        return False

import praw
count=0
reddit = praw.Reddit(
    client_id = "YOUR CLIENT ID",
    client_secret = "YOUR CLIENT SECRET",
    user_agent = "YOUR USER AGENT",
    user_name = "YOUR USERNAME"
    password = "YOUR PASSWORD"
)

def Perform(subreddit_list):
    subreddits =[]
    for sub in subreddit_list:
        subreddits.append(reddit.subreddit(sub))
    posts = []
    urls = []
    post_data = []
    total_score=0
    count=0
    for s in subreddits:
        for submission in s.hot():
            for comment in submission.comments:
                # print(comment.body)
                if(IsDepressed(comment.body)):
                    comment.reply("Here's a hug ⊂(◉‿◉)つ")                    
Perform(["India", "Nepal", "Pune"], 17)