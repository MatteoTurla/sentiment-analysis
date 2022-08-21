import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers

class DistillBert:
    
    def __init__(self):
        
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        
    def transform(self, df, batch_size=512):
        batches_features = []
        N = len(df)
        
        for i, batch_idx, in enumerate(range(0, N, batch_size)):
            batch_df = df.iloc[batch_idx:batch_idx+batch_size]
            
            features = self.transform_batch(batch_df)
            batches_features.append(features)
            
            
            print(f'batch number {i} out of {N//batch_size} done!')
            
        return np.concatenate(batches_features)
    
    def transform_batch(self, batch):
        tokenized = batch.text.apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
        
        # pad batch to same lenght
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        
        input_tensor = torch.tensor(padded)  
        mask_tensor = torch.tensor(attention_mask)
        
        # extract features
        with torch.no_grad():
            last_hidden_states = self.model(input_tensor, attention_mask=mask_tensor)
            features = last_hidden_states[0][:,0,:].numpy()
        
        return features
