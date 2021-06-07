import torch.nn as nn
import torch
from torchcrf import CRF
from dataset import labels_dict
from transformers import AutoModel, AutoTokenizer



tokenizers_dict = {
    'deeppavlov_bert': 'DeepPavlov/rubert-base-cased',
}

pretrained_models_dict = {
    'deeppavlov_bert': ('DeepPavlov/rubert-base-cased', 768),
}


class BERT_LSTM(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(BERT_LSTM, self).__init__()
        self.output_dim = len(labels_dict)
        self.bert = AutoModel.from_pretrained(pretrained_models_dict[pretrained_model][0])
        bert_dim = pretrained_models_dict[pretrained_model][1]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(labels_dict))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0]) 
        x = self.bert(x, attention_mask=attn_masks)[0]
        #print('bert', x)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        #print('lstm', x)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        #print('linear', x)
        return x
    
    
class BERT_LSTM_CRF(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(BERT_LSTM_CRF, self).__init__()
        self.bert = BERT_LSTM(pretrained_model, freeze_bert, lstm_dim)
        self.crf = CRF(len(labels_dict), batch_first=True)

    def log_likelihood(self, x, attn_masks, y):
        x = self.bert(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, x, attn_masks, y):
        #print("X_SHAPE", x.shape)
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0]) 
        
        x = self.bert(x, attn_masks)
        #print("X_SHAPE_BERT", x.shape)
        attn_masks = attn_masks.byte()
        #print("MASK_SHAPE", attn_masks.shape)
        dec_out = self.crf.decode(x, mask=attn_masks)
        y_pred = torch.zeros(y.shape).long().to(y.device)
        #print('SHAPE', len(dec_out), 'SHAPE0', len(dec_out[0]))
        #print(y_pred.shape)
        #print("Y_SHAPE", y[0].shape)
        for i in range(len(dec_out)):
            y_pred[i, :len(dec_out[i])] = torch.tensor(dec_out[i]).to(y.device)
        return y_pred