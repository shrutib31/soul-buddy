import torch
import torch.nn as nn
from transformers import AutoModel

class SoulBuddyClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        # Base transformer
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # Situation head (8 classes)
        self.situation_head = nn.Linear(hidden_size, 8)
        
        # Severity head (3 classes)
        self.severity_head = nn.Linear(hidden_size, 3)
        
        # Intent head (8 classes)
        self.intent_head = nn.Linear(hidden_size, 8)
        
        # Risk head (binary)
        self.risk_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0]

        situation_logits = self.situation_head(pooled_output)
        severity_logits = self.severity_head(pooled_output)
        intent_logits = self.intent_head(pooled_output)
        risk_logits = self.risk_head(pooled_output)

        return situation_logits, severity_logits, intent_logits, risk_logits
