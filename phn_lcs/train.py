import warnings
import numpy as np
import torch
import torch.nn as nn
import os
os.environ["SAFETENSORS_AVAILABLE"] = "0"
from transformers import PreTrainedTokenizerFast, T5EncoderModel, T5Config, Trainer, TrainingArguments, PreTrainedTokenizer
import os
import json
import logging
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CMU pho2dict
phoneme_to_id = {
    "AA": 0, "AE": 1, "AH": 2, "AO": 3, "AW": 4, "AY": 5,
    "B": 6, "CH": 7, "D": 8, "DH": 9, "EH": 10, "ER": 11,
    "EY": 12, "F": 13, "G": 14, "HH": 15, "IH": 16, "IY": 17,
    "JH": 18, "K": 19, "L": 20, "M": 21, "N": 22, "NG": 23,
    "OW": 24, "OY": 25, "P": 26, "R": 27, "S": 28, "SH": 29,
    "T": 30, "TH": 31, "UH": 32, "UW": 33, "V": 34, "W": 35,
    "Y": 36, "Z": 37, "ZH": 38,
    "<pad>": 39, "<unk>": 40, "<cls>": 41, "<sep>": 42
}

class PhonemeTokenizer(PreTrainedTokenizer):
    def __init__(self, phoneme_to_id, **kwargs):
        self.phoneme_to_id = phoneme_to_id
        super().__init__(**kwargs)
        self.id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.cls_token = "<cls>"
        self.sep_token = "<sep>"
        
    def get_vocab(self):
        # return vocabulary table
        return self.phoneme_to_id
        
    def _convert_token_to_id(self, token):
        # Convert a single phoneme to its ID
        return self.phoneme_to_id.get(token, self.phoneme_to_id.get(self.unk_token))

    def _convert_id_to_token(self, index):
        # Convert a single ID back to a phoneme
        return self.id_to_phoneme.get(index, self.unk_token)

    def _tokenize(self, text):
        # Split text into phonemes and map them to IDs
        return [self.phoneme_to_id.get(phoneme, self.phoneme_to_id.get(self.unk_token)) for phoneme in text.split()]

    def encode(self, text, max_length = 120, add_special_tokens = True, padding = True):
        max_len = max_length
        token_ids = self._tokenize(text)
        if add_special_tokens:
            token_ids = token_ids + [self.phoneme_to_id[self.sep_token]]
        if padding:
            prev_len = len(token_ids)
            token_ids = token_ids + [self.phoneme_to_id[self.pad_token]] * (max_len - prev_len)
            mask = [1] * prev_len + [0] * (max_len - prev_len)
        if padding:
            return {"input_ids": torch.tensor(token_ids), "attention_mask": torch.tensor(mask)}
        else:
            return {"input_ids": torch.tensor(token_ids), "attention_mask": None}
                

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.id_to_phoneme[token_id] for token_id in token_ids if token_id in self.id_to_phoneme]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [self.pad_token, self.cls_token, self.sep_token]]
        return " ".join(tokens)

    def __len__(self):
        return len(self.phoneme_to_id)

tokenizer = PhonemeTokenizer(phoneme_to_id)

class PhonemeDataset(Dataset):
    def __init__(self, json_path, phoneme_to_id, max_len=120):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = PhonemeTokenizer(phoneme_to_id = phoneme_to_id)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _encode(self, phoneme_sequence):
        return self.tokenizer.encode(phoneme_sequence, max_length = self.max_len, add_special_tokens = True)

    def __getitem__(self, idx):
        item = self.data[idx]
        ref = self._encode(item["REF"])["input_ids"].clone().detach()
        ref_mask = self._encode(item["REF"])["attention_mask"].clone().detach()
        src = self._encode(item["SRC"])["input_ids"].clone().detach()
        src_mask = self._encode(item["SRC"])["attention_mask"].clone().detach()
        label = list(map(int, item["LABEL"].split()))
        label = label[:self.max_len] + [-1] * (self.max_len - len(label))
        return {
            "ref": ref,
            "src": src,
            "ref_mask": ref_mask,
            "src_mask": src_mask,
            "label": torch.tensor(label, dtype=torch.long)  # label 不是张量，所以保留 torch.tensor
        }

json_path = "github/phn/data_phn/example.json"  # choose your own data, it is just an example
dataset = PhonemeDataset(json_path, phoneme_to_id)

train_size = int(0.97 * len(dataset))  
test_size = len(dataset) - train_size  

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print("Dataset is ready...")

def collate_fn(batch):
    ref = torch.stack([item['ref'] for item in batch])
    src = torch.stack([item['src'] for item in batch])
    ref_mask = torch.stack([item['ref_mask'] for item in batch])
    src_mask = torch.stack([item['src_mask'] for item in batch])
    label = torch.stack([item['label'] for item in batch])
    return {
        'ref': ref,
        'src': src,
        'ref_mask': ref_mask,
        'src_mask': src_mask,
        'label': label
    }

class PhonemeBoundaryAlignerT5(nn.Module):
    def __init__(self, pretrained_model_name="t5-small", phoneme_vocab_size=50, hidden_dim=512, num_filters = 16):
        super(PhonemeBoundaryAlignerT5, self).__init__()
        # Load pretrained T5 model
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_name)
        
        # Resize token embeddings to fit phoneme vocab size
        self.encoder.resize_token_embeddings(phoneme_vocab_size)

        # 1D CNN Layer
        self.conv1d = nn.Conv1d(
            in_channels = hidden_dim * 2,
            out_channels = num_filters,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
                
        # Boundary Predictor: Fully connected layers
        self.MLP = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_filters, num_filters//2),
            nn.ReLU(),
            nn.Linear(num_filters//2, 4),
        )
        
        # Re-initialize weights if needed
        self._init_weights()

    def forward(self, ref, src, ref_mask=None, src_mask=None):
        # Encode reference and source phonemes using T5
        ref_output = self.encoder(input_ids=ref, attention_mask=ref_mask).last_hidden_state  # (batch_size, src_len, hidden_dim)
        src_output = self.encoder(input_ids=src, attention_mask=src_mask).last_hidden_state  # (batch_size, src_len, hidden_dim)
        
        # Combine ref and src features
        alignment_features = torch.cat((ref_output, src_output), dim=-1)  # (batch_size, src_len, hidden_dim * 2)
        alignment_features = alignment_features.permute(0, 2, 1)                               # (batch_size, hidden_dim * 2, src_len)
        
        # print(alignment_features.shape)
        
        # Process with Conv layer to capture contextual information
        conv_features = self.conv1d(alignment_features)     # (batch_size, num_filters, src_len) 
        conv_features = conv_features.permute(0, 2, 1)                      # (batch_size, src_len, num_filters)
        
        # Through MLP layer
        mlp_features = self.MLP(conv_features) # (batch_size, src_len, 3)
        
        return mlp_features
    
    def _init_weights(self):
        """
        Reinitialize model parameters for layers other than embeddings.
        """
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        # Apply custom weight initialization to layers
        self.apply(init_weights)

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.2, 0.08, 0.2, 0.2], gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)  # (batch_size, src_len, num_classes)

        valid_mask = (targets != -1).float()
        safe_targets = torch.where(targets == -1, torch.zeros_like(targets), targets)

        class_probs = probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)  # (batch_size, src_len)
        ce_loss = -torch.log(class_probs)  # (batch_size, src_len)
        focal_loss = (1 - class_probs) ** self.gamma * ce_loss  # (batch_size, src_len)

        focal_loss = focal_loss * valid_mask
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                self.alpha = torch.tensor(self.alpha, device=logits.device)
            alpha = self.alpha.gather(0, safe_targets.view(-1)).view_as(safe_targets)  # (batch_size, src_len)
            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.sum() / valid_mask.sum()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ref = inputs['ref']
        src = inputs['src']
        ref_mask = inputs['ref_mask']
        src_mask = inputs['src_mask']
        label = inputs['label']
        
        outputs = model(ref, src, ref_mask, src_mask)
        loss = criterion(outputs, label.long())
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    warmup_steps=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=10,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_safetensors=False
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model = PhonemeBoundaryAlignerT5(phoneme_vocab_size=len(tokenizer)).to(device)
criterion = FocalLoss(reduction='mean')

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
)

print("Start training...")
trainer.train()

trainer.save_model('github/phn/model/phn_lcs_1.pth')