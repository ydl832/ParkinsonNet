# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:01:38 2022
@author: Yande
"""
import torch
import clip
output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_clip():
    model_clip, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    text_encoder = model_clip.encode_text
    text_token = clip.tokenize
    return text_encoder, text_token

class TextEncoder(nn.Module):
   def __init__(self):
       super(TextEncoder, self).__init__()
       self.text_encoder, self.text_token = feature_clip()
       self.fc = nn.Linear(1600, 7)
       
   def forward(self)
       text = ["Normal","Slight","Mild","Moderate or Severe"]
       
       token_text = self.text_token(text).to(device)
       text_feature = self.fc(self.text_encoder(token_text).type(torch.FloatTensor).cuda()

       return text_feature

def text_encoder():
    feature = TextEncoder()
    return feature
