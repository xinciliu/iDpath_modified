# iDPath
iDpath is the model came up by Researchers in paper Yang, J.†. , Li, Z. , Wu, W.K.K. , Yu, S. , Xu, Z.†. , Chu, Q.*. & Zhang, Q.*. (Nov 2022). Deep Learning Identifies Explainable Reasoning Paths of Mechanism of Action for Drug Repurposing from Multilayer Biological Network. Briefings in Bioinformatics. 23/6. bbac469 doi:10.1093/bib/bbac469 

The github link is : https://github.com/JasonJYang/iDPath . 

# Modifications
I did some modifications to model/model.py and try different deep learning models other than LSTM.

I did some modifications to model/loss.py and try different loss functions to see the performance. 

# Introductions for use
There are four different deep-learning models you can try:

LSTM: lstm   BILSTM: bilstm  RNN: rnn BIRNN: birnn

You can also choice different loss functions:

nll_loss: nll_loss   binary_cross_entropy_with_logits: bce_withlogits_loss     softmax_cross_entropy_with_logits: sce_withlogits_loss

   # generating the config function using different deep-learning models/loss functions
   
     you can use:  python generate_config.py ${model_name} ${loss_name} ${config_file_name}  
     
     to generate the training config file for yourself
     
     for example: python generate_config.py bilstm softmax_cross_entropy_with_logits config_file1
     
                  "your config file is saved at config/config_file1.json"
                  
   # Training with the model you choice for yourself:
   
     python train.py --config config/config_file1.json --resume saved/models/iDPath-main/0117_164440/model_best.pth
