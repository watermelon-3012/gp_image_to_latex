# LaTeX Generation for Mathematical Expressions: An experiment with different models

In this work, we experiment with different models for the problem of generating LaTeX code for Mathematical Expressions. The models considered in this paper all use the Encoder-Decoder architecture: CNN-LSTM, pure Vision Transformer (ViT), and Convolutional Transformer. After training, we evaluate these three models on two metrics: Normalized Levenshtein Distance and BLEU Score. Experimental results show that the Convolutional Transformer model consistently outperforms both CNN–LSTM and ViT, demonstrating good generalization and ability of generating LaTeX code from images of mathematical expressions. Based on these findings, we discuss the strengths and limitations of each architecture. Finally, the trained models are deployed on a web-based platform, allowing users to interact with and experiment with the systems.

**Key Words:** Natural Language Processing, LaTeX Code Generator, LaTeX dictionary, Transformer-based Models, Hybrid Models, Text Generation Evaluation, UI/UX Implementation

The repository includes the following files:
- README.md – Introduction of the project
- all_in_one_experiment.ipynb – Contains everything from data preprocessing, model implementation and performance evaluation, and metrics.
- Web Implementaion – The folder containing source code for the web implementation.
- G32-GP2026-ICT.pdf – The report for this project

**Group Members:**
1. Nguyen Quang Minh - minhnq.22ba13221@usth.edu.vn
2. Nguyen Cao Manh Thang - thangncm.23bi14403@usth.edu.vn
4. Le Sy Han - hanls.23bi14150@usth.edu.vn
5. Le Hoang Dat - datlh.23bi14087@usth.edu.vn
6. Bui Tuan Thanh - thanhbt.22ba13284@usth.edu.vn
7. Duong Duc Anh - anhdd.23bi14012@usth.edu.vn

