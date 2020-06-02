from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM
from transformers.modeling_bert import BertOnlyMLMHead
from pathlib import Path
import torch
from torch import nn
from parallel import DataParallelCriterion
import sys


class BertClass(BertPreTrainedModel):

    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mlm_probability = 0.15
        self.bert = BertModel(config)
        self.tokenizer = tokenizer
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.topic_cls = nn.Linear(config.hidden_size, config.num_labels)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.context_emb = nn.Parameter(torch.Tensor(config.hidden_size).normal_(mean=0.0, std=config.initializer_range))
        self.activation = nn.Tanh()
        self.init_weights()
        
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def topic_seeds(self, seed_word_list):
#         self.topic_seeds = seed_word_list
        # -100 tokens will be masked during topic prediction
        self.word_class = -100 * torch.ones(self.config.vocab_size, dtype=torch.long)
        for i, topic_words in enumerate(seed_word_list):
            for word in topic_words:
                self.word_class[word] = i
    
    def set_mask_id(self, mask_id):
        self.mask_id = mask_id
        
    def mask_topic(self, inputs):
        labels = self.word_class[inputs]
        inputs[labels != -100] = self.mask_id
        
        return inputs, labels
    
    def propagate_topic(self, inputs):
        labels = self.word_class[inputs]
        out_label = -100 * torch.ones(inputs.shape[0], dtype=torch.long)
        for i, label in enumerate(labels):
            if len(label[label != -100]) > 0:
                out_label[i] = label[label != -100][0]
        
        return inputs, out_label
    
    def attn(self, attn_mask, last_hidden_state):
        transformed_state = self.dense(last_hidden_state)
        transformed_state = self.activation(transformed_state)
        attn_scores = torch.mul(transformed_state, self.context_emb)
        attn_scores = torch.sum(attn_scores, dim=-1)
        attn_scores = attn_scores + attn_mask
        attn_weights = nn.Softmax(dim=-1)(attn_scores)
        seq_emb = torch.bmm(attn_weights.unsqueeze(1), last_hidden_state).squeeze(1)
        
        return seq_emb, attn_weights
    
    def get_cls_attn_mask(self, input_ids):
        attn_mask = input_ids.clone().fill_(1.0)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                               for val in input_ids.tolist()]
        attn_mask.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        attn_mask.masked_fill_(input_ids == self.tokenizer.pad_token_id, value=0.0)
        attn_mask = (1.0 - attn_mask) * -10000.0
        
        return attn_mask
    
    def forward(self, input_ids, pred_mode, cls_attn_mask=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, with_attn=False, with_states=False):
        bert_outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)
        last_hidden_state = bert_outputs[0]
        
        if pred_mode == "topic":
            logits = self.topic_cls(last_hidden_state)
        elif pred_mode == "token":
            logits = self.cls(last_hidden_state)
        else:
            sys.exit("Wrong pred_mode!")
        
        if cls_attn_mask is not None:
            seq_emb, attn_weights = self.attn(cls_attn_mask, last_hidden_state)
            logits = self.topic_cls(seq_emb)
        
        outputs = (logits,)
        
        if with_attn:
            outputs = outputs + (attn_weights,)
        if with_states:
            outputs = outputs + (last_hidden_state,)
        
        outputs = outputs + bert_outputs[1:]
        
        return outputs
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
