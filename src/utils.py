# import spacy

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import numpy as np

import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class Instance:
    """
    Represents a shallow wrapper of tokens
    """
    tokens_i: list[int]
    tokens_o: list[int] 
    tokens_o_true: list[int]
    
    def __iter__(self):
        return iter([self.tokens_i, self.tokens_o, self.tokens_o_true])



class NLPDataset(Dataset):
    """Abstracts data which will be used by our models. Instances of NLPDataset will be passed to DataLoader.
    
    Implements __getitem__ and __len__."""
    
    def __init__(self, 
                 path_to_data_en: str,
                 path_to_data_de: str,
                 max_dataset_entries: int = -1,
                 max_size = -1, 
                 min_freq = -1, 
                 vocab_en: 'Vocab' = None,
                 vocab_de: 'Vocab' = None) -> None:
        super().__init__()
        
        frequencies_en = {}
        frequencies_de = {}
        instances = []
        
        # open files and read instances
        with open(path_to_data_en, 'r') as f_en, open(path_to_data_de, 'r', encoding='utf-8') as f_de:
            sentences_en = f_en.read().strip().lower().replace('.', '').replace(',', '').split('\n')
            sentences_de = f_de.read().strip().lower().replace('.', '').replace(',', '').split('\n')
            
            assert len(sentences_en) == len(sentences_de), "Number of sentences must be equal!"
            
            counter = 0
            for sentence_en, sentence_de in zip(sentences_en, sentences_de):
                
                # limiter
                if counter == max_dataset_entries:
                    break
                counter += 1
                
                # split by whitespace
                words_en = split_into_ngrams(sentence_en) #sentence_en.split(" ") #list(nlp_en(sentence_en)) 
                words_de = split_into_ngrams(sentence_de) #sentence_de.split(" ") #list(nlp_de(sentence_de)) 
                
                # save freqs and embeddings EN
                for w in words_en:
                    if str(w) in frequencies_en:
                        frequencies_en[str(w)] += 1
                    else:
                        frequencies_en[str(w)] = 1
                
                # save freqs and embeddings DE
                for w in words_de:
                    if str(w) in frequencies_de:
                        frequencies_de[str(w)] += 1
                    else:
                        frequencies_de[str(w)] = 1
                
                # save
                words_en = list(str(w) for w in words_en)
                words_de = list(str(w) for w in words_de)
                instances.append((words_en, words_de))
            
            self.frequencies_en = frequencies_en
            self.frequencies_de = frequencies_de
            self.instances = instances
        
        # build vocab 
        if vocab_en is not None:
            self.vocab_en = vocab_en
            self.vocab_de = vocab_de
        else:
            self.vocab_en = Vocab(frequencies_en, max_size, min_freq)
            self.vocab_de = Vocab(frequencies_de, max_size, min_freq)
            
    def __getitem__(self, index) -> Instance:
        
        # fetch
        tokens_en = ["<SOS>"] + self.instances[index][0] + ["<EOS>"]
        tokens_de_input = ["<SOS>"] + self.instances[index][1]
        tokens_de_output = self.instances[index][1] + ["<EOS>"]
        
        # transform
        tokens_en = self.vocab_en.encode(tokens_en)
        tokens_de_input = self.vocab_de.encode(tokens_de_input)
        tokens_de_output = self.vocab_de.encode(tokens_de_output)
        
        return Instance(tokens_en, tokens_de_input, tokens_de_output)
    
    def __len__(self):
        return len(self.instances)

class Vocab:
    """Represents an NLP vocabulary. Can encode ('string-to-int' operation) a list of strings."""
    
    def __init__(self, frequencies: dict, max_size = -1, min_freq = -1) -> None:
        
        # tokens sorted by freqs
        sorted_freqs = sorted(frequencies, key=frequencies.get, reverse=True)
        
        # check and apply min_freq filter
        if min_freq > -1:
            sorted_freqs = list(filter(lambda x: frequencies[x] > min_freq, sorted_freqs))
        
        # check and apply max_size filter
        if max_size > -1:
            if max_size < len(sorted_freqs):
                sorted_freqs = sorted_freqs[:max_size]
        
        # build stoi dict
        self.stoi: dict[str, int] = {}
        
        self.stoi["<PAD>"] = 0
        self.stoi["<UNK>"] = 1
        self.stoi["<SOS>"] = 2
        self.stoi["<EOS>"] = 3
        
        i = 4
        for token in sorted_freqs:
            self.stoi[token] = i
            i += 1

        # build itos
        self.itos: dict[int, str] = {}
        for key in self.stoi:
            self.itos[self.stoi[key]] = key
    
    def encode(self, tokens: list[str]) -> list[int]:
        """Encodes a list of strings (tokens) into a list of ints."""
        
        output = []
        tmp = 0
        
        for token in tokens:
            if token in self.stoi:
                tmp = self.stoi[token]
            else:
                tmp = self.stoi["<UNK>"]
            output.append(tmp)
            
        return output
    
    def decode(self, encoded: list[int]):
        return list([self.itos[i] for i in encoded])

def create_look_ahead_mask(L, S):
    return torch.triu(torch.ones(L, S), diagonal=1).bool()

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

def split_into_ngrams(sentence, n = -1):
    assert n > 0 or n == -1

    if n == -1:
        return sentence.split(" ")
    else:
        return [sentence[i:i+n] for i in range(len(sentence)-n+1)]

def get_accuracy(model, dl):
    
    with torch.no_grad():
        
        model.eval()
        
        n_correct = 0
        n_total = 0
        for i_batch, (input_texts, input_masks, output_texts, y_true, attention_masks, key_padding_masks) in enumerate(dl):            
            
            # get predictions
            y_pred = model.forward(input_texts, input_masks, output_texts, attention_masks, key_padding_masks)
            
            # reshape, argmax and convert to numpy
            y_pred = torch.argmax(y_pred.reshape(-1, model.d_output), axis=1).cpu().numpy()
            y_true = torch.argmax(y_true.reshape(-1, model.d_output), axis=1).cpu().numpy()
            
            # filter PAD values
            y_pred_filtered = []
            y_true_filtered = []
            for i in range(len(y_true)):
                if y_true[i] != 0:
                    y_pred_filtered.append(y_pred[i])
                    y_true_filtered.append(y_true[i])
            
            y_pred = np.array(y_pred_filtered)
            y_true = np.array(y_true_filtered)
            
            # add to n_correct and n_total
            n_correct += sum(y_pred == y_true)
            n_total += len(y_pred)
        
        # return accuracy
        return n_correct/n_total

def translate(model, english_text, input_vocab, output_vocab):        
        # tokenize
        english_text = english_text.strip().lower().replace('.', '').replace(',', '')
        tokens_en = ["<SOS>"] + english_text.split(" ") + ["<EOS>"]
        
        # stoi en
        tokens_en = torch.tensor(input_vocab.encode(tokens_en)).reshape(1, -1)
        
        # stoi de
        translation_output = output_vocab.encode(["<SOS>"]) # encoded <SOS>
        tokens_de = torch.tensor(translation_output).reshape(1, -1)
        
        # feed to network
        with torch.no_grad():
            while True:
                # get last simbol
                res = model.forward(tokens_en, None, tokens_de, None, None)[0]
                
                new_simbol = torch.argmax(res[-1]).item()
                
                # add
                translation_output.append(new_simbol)
                if new_simbol == 3 or len(translation_output) > 20:
                    break
                
                tokens_de = torch.tensor(translation_output).reshape(1, -1)
            
            print("Final output: " + ' '.join(output_vocab.decode(translation_output)))