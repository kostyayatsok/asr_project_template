import logging
from typing import List
import re
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    if len(dataset_items) == 0: return {}
    result_batch = {
        'spectrogram' : [],
        'spectrogram_length' : [],
        'text_encoded' : [],
        'text_encoded_length' : [],
        'text' : []
    }
    for i in range(len(dataset_items)):
        result_batch['spectrogram'].append(dataset_items[i]['spectrogram'].T)
        result_batch["spectrogram_length"].append(result_batch['spectrogram'][-1].shape[0])
        result_batch['text_encoded'].append(dataset_items[i]['text_encoded'].T)
        result_batch['text_encoded_length'].append(result_batch['text_encoded'][-1].shape[0])
        result_batch['text'].append(re.sub(r'[^\w\s]','',dataset_items[i]['text']))
    
    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], padding_value=-1, batch_first=True)
    result_batch['spectrogram'] = result_batch['spectrogram'][:,:,:,0]
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])
    
    result_batch['text_encoded'] = pad_sequence(result_batch['text_encoded'], padding_value=-1, batch_first=True).long()
    result_batch['text_encoded'] = result_batch['text_encoded'][:,:,0]
    
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    


    return result_batch