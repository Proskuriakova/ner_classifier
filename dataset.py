
import torch
import numpy as np
from transformers import AutoTokenizer



# специальные токены для энкодеров
tokens_ids = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# словарь таргетов
# labels_dict = {
#     'o-o': 0, 'o-dot': 1, 'o-comma': 2, 'o-question': 3, 'o-exclamation': 4, 
#     'caps-o': 5, 'caps-dot': 6, 'caps-comma': 7, 'caps-question': 8, 'caps-exclamation': 9} 

labels_dict = {
    'o-o': 0, 'o-dot': 1, 'o-comma': 2, 
    'caps-o': 3, 'caps-dot': 4, 'caps-comma': 5} 

# датасет класс
# files - файл или лист из файлов в формате txt, с разметкой Conll
# tokenizer - токенайзер 
# sequence_len -  длина последовательности

class NER_Dataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, sequence_len, token_style):
        if isinstance(files, list):
            self.data = []
            for file in files:
                self.data += parse_data(file, tokenizer, sequence_len, token_style)
        else:
            self.data = parse_data(files, tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.token_style = token_style


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask
    

# вспомогательная функция для сборки датасета для обучения
# file_path -  путь до файла
# tokenizer
# sequence_len - длина последовательности
# token_style - для разметки специальными токенами: начало последовательности, конец, паддинги
# возвращает [x, y, attn_mask, y_mask]
# x - размеченная специальными токенами последовальность
# y - таргет
# attn_mask - маска для игнорирования специальных токенов 
# y_mask

def parse_data(file_path, tokenizer, sequence_len, token_style):
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().split('\n') if line.strip()]
        idx = 0
        while idx < len(lines):
            x = [tokens_ids[token_style]['START_SEQ']]
            y = [0]
            y_mask = [1]  

            while len(x) < sequence_len - 1 and idx < len(lines):
                word, punc = lines[idx].split(' ')
                tokens = tokenizer.tokenize(word)
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)
                    if len(tokens) > 0:
                        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(labels_dict[punc])
                    y_mask.append(1)
                    idx += 1
            x.append(tokens_ids[token_style]['END_SEQ'])
            y.append(0)
            y_mask.append(1)
            if len(x) < sequence_len:
                x = x + [tokens_ids[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y = y + [0 for _ in range(sequence_len - len(y))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != tokens_ids[token_style]['PAD'] else 0 for token in x]
            data_items.append([x, y, attn_mask, y_mask])
    return data_items