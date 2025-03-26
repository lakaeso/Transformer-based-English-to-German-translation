import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils import NLPDataset, create_look_ahead_mask, dense_to_one_hot, get_accuracy

from model import Transformer

# hiperparams
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPOCH = 100

BATCH_SIZE_TRAIN = 50

BATCH_SIZE_VALID = 50

MODEL_DIM = 96

ENCODER_STACK_LEN = 4

DECODER_STACK_LEN = 8

NUM_HEADS = 8

# init train and validation datasets
train_data = NLPDataset(
    './datasets/multi30k-dataset/train.en', 
    './datasets/multi30k-dataset/train.de',
    max_size=-1,
    min_freq=3
)

valid_data = NLPDataset(
    './datasets/multi30k-dataset/val.en', 
    './datasets/multi30k-dataset/val.de',
    max_size=-1,
    min_freq=-1,
    vocab_en = train_data.vocab_en,
    vocab_de = train_data.vocab_de
)

# input and target vocab size
input_vocab_size = len(train_data.vocab_en.stoi)
target_vocab_size = len(train_data.vocab_de.stoi)


def pad_collate_fn(batch, pad_index=0, target_vocab_size=target_vocab_size):
    """Pads and converts input data to tensors and also adds info about their respective lengths.
    
    Returns a triplet in a form of (input_texts, output_texts)."""
    
    # get input and output texts
    input_texts, output_texts, output_texts_true = zip(*batch)
    
    # lengths
    lengths_input = list([len(txt) for txt in input_texts])
    lengths_output = list([len(txt) for txt in output_texts])
        
    # pad input text
    input_texts = list([torch.tensor(text) for text in input_texts])    
    input_texts_tensor = torch.nn.utils.rnn.pad_sequence(input_texts, batch_first=True, padding_value=pad_index)
    
    # input mask
    max_input_len = max(lengths_input)
    input_masks = []
    for l in lengths_input:
        input_masks.append([False] * l + [True] * (max_input_len - l))
    input_masks_tensor = torch.tensor(input_masks, dtype=torch.bool)
    
    # pad output text
    output_texts = list([torch.tensor(text) for text in output_texts])
    output_texts_tensor = torch.nn.utils.rnn.pad_sequence(output_texts, batch_first=True, padding_value=pad_index)
    
    # pad output text true
    output_texts_true = list([torch.tensor(text) for text in output_texts_true])
    output_texts_true_tensor = torch.nn.utils.rnn.pad_sequence(output_texts_true, batch_first=True, padding_value=pad_index)

    # prepare y_true... 
    y_true = []
    for i in range(len(output_texts_true)):
        y_true.append(torch.tensor(dense_to_one_hot(output_texts_true_tensor[i], target_vocab_size), dtype=torch.float32))
    y_true = torch.stack(y_true)
    
    longest_output_text_len = output_texts_tensor.shape[1]
    
    # attention masks
    attn_masks_tensor = create_look_ahead_mask(longest_output_text_len, longest_output_text_len)
    
    # key padding masks
    key_padding_masks = []
    for l in lengths_output:
        key_padding_masks.append([False] * l + [True] * (longest_output_text_len - l))
    key_padding_masks = torch.tensor(key_padding_masks, dtype=torch.bool)
    
    return input_texts_tensor.to(DEVICE), input_masks_tensor.to(DEVICE), output_texts_tensor.to(DEVICE), y_true.to(DEVICE), attn_masks_tensor.to(DEVICE), key_padding_masks.to(DEVICE)


# init data loaders
train_dl = DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True, collate_fn=pad_collate_fn, num_workers=8, persistent_workers=True)
valid_dl = DataLoader(valid_data, batch_size=BATCH_SIZE_VALID, shuffle=True, collate_fn=pad_collate_fn, num_workers=2, persistent_workers=True)

# init transformer model
model = Transformer(
    MODEL_DIM,
    input_vocab_size,
    target_vocab_size,
    ENCODER_STACK_LEN,
    DECODER_STACK_LEN,
    NUM_HEADS
).to(DEVICE)

# criterion and optim
criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.get_parameters())

if __name__ == '__main__':

    # train loop
    for i_epoch in range(1, MAX_EPOCH + 1):
        
        for i_batch, (input_texts, input_masks, output_texts, y_true, attention_masks, key_padding_masks) in enumerate(train_dl):
            
            # set model in train mode
            model.train()
            
            # set zero grad
            optim.zero_grad()
            
            # get model outputs
            y_pred = model(input_texts, input_masks, output_texts, attention_masks, key_padding_masks)
            
            # take only predicitons
            y_pred = y_pred.reshape(-1, model.d_output)
            
            # take only true values
            y_true = y_true.reshape(-1, model.d_output)
            
            # compute loss
            loss = criterion(y_pred, y_true)
            
            # eval model
            if i_batch % 100 == 0:
                model.eval()
                print(f"epoch = {i_epoch}, batch = {i_batch}, eval acc = {get_accuracy(model, valid_dl)*100:.2f}%")
                model.train()    
            
            # bwd pass
            loss.backward()
            
            # optim step
            optim.step()