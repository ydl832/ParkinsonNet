# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:01:38 2022
@author: Yande
"""
import torch
import clip
import torch.nn as nn
import os
    

def feature_clip():
    model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = model.encode_text
    text_token = clip.tokenize
    return text_encoder, text_token

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder, self.text_token = feature_clip()
        self.fc = nn.Linear(512, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self,text):
    
        token_text = self.text_token(text).to(self.device)
        with torch.no_grad():
            text_feature = self.text_encoder(token_text).type(torch.FloatTensor).cuda()

        text_feature = self.relu(self.bn(self.fc(text_feature)))
        #torch.save(text_feature, '/home/yande/ST-TR/code/checkpoints/text_feature.pt')
        return text_feature

        