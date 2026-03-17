# LaTeX Generation for Mathematical Expressions: An experiment with different models
This project explores multiple deep learning approaches for generating LaTeX code from images of mathematical expressions.

## Overview
We experiment with three encoder–decoder architectures:
- CNN–LSTM  
- Vision Transformer (ViT)  
- Convolutional Transformer  
All models are trained end-to-end to convert mathematical expression images into LaTeX sequences.

## Evaluation
We evaluate model performance using:
- Normalized Levenshtein Distance  
- BLEU Score  

### Key Findings
- The **Convolutional Transformer** consistently outperforms:
  - CNN–LSTM  
  - Vision Transformer (ViT)  
- It shows better **generalization** and **robustness** in LaTeX generation.

## Deployment
The trained models are deployed on a **web-based interface**, allowing users to:
- Upload images of mathematical expressions.
- Generate LaTeX code interactively with 3 models.

## Features
- Multiple model architectures for comparison  
- End-to-end image-to-LaTeX pipeline  
- Quantitative evaluation metrics  
- Interactive demo (web UI)  

## Tech Stack
- PyTorch  
- Computer Vision  
- Seq2Seq (Sequence-to-Sequence Learning)  
- Transformer-based architectures  
- Attention Mechanisms  
- Image-to-Text Modeling
- Tokenization (LaTeX vocabulary)
- Beam Search Decoding

## Team Members
- Nguyen Quang Minh – minhnq.22ba13221@usth.edu.vn  
- Nguyen Cao Manh Thang – thangncm.23bi14403@usth.edu.vn  
- **Le Sy Han – hanls.23bi14150@usth.edu.vn**
- Le Hoang Dat – datlh.23bi14087@usth.edu.vn  
- Bui Tuan Thanh – thanhbt.22ba13284@usth.edu.vn  
- Duong Duc Anh – anhdd.23bi14012@usth.edu.vn  