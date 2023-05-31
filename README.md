# CL-Robustness
Our goal is to understand whether there are differences in how contrastive learning learns the representation from data compared to supervised learning from the adversarial perspective. To this end, we conduct extensive experiments to evaluate the robustness of the following learning schemes:

- Contrastive Learning (CL): Within the standard framework of SimCLR, contrastive learning trains a base encoder by minimizing a contrastive loss over the representations projected into a latent space (Figure 1(a)). The extracted features will be employed to train a linear classifier on a downstream task as shown in Figure 1(a).
- Supervised Contrastive Learning (SCL): A supervised extension of contrastive learning  to avoid false positive pairs selection by leveraging the label information.
- Supervised Learning (SL): The network consists of a base encoder followed by a fully connected layer as a linear classifier, as shown in figure 1(b). In this case, cross entropy between the true and predicted labels is utilized for training the network parameters.




## An overview of the methodology
<p align="center">
<img src="./figures/contrastive.jpg" alt="Alt Text" width="500">
 <br>
  Figure 1(a) Contrastive and Supervised Contrastive Learning.
</p>

<p align="center">
<img src="./figures/supervised.jpg" alt="Alt Text" width="500">
  <br>
  Figure 1(b) Supervised Learning.
</p>

<div align="center">

| Scenarios   | Pretraining Phase    | Finetuning Phase          |
|-------------|----------------------|---------------------------|
| ST          | Standard Training    | Standard Training         |
| AT          | Adversarial Training | Standard Training         |
| Partial-AT  | Adversarial Training | Partial Adversarial Training |
| Full-AT     | Adversarial Training | Full Adversarial Training   |

Caption: Summary of the training scenarios.

</div>

## TSNE Visualization

<img src="./figures/tsne.PNG" alt="Different Scenarios for Training" width="500" height="400">


###### Semi-supervised learning schemes (SL-CL and SCL-CL) separate classes much more clearly than contrastive learning (CL) scheme.

## Representation Visualization
<img src="./figures/ST_ST.PNG" alt="Different Scenarios for Training" width="830" height="230">
<img src="./figures/CL.PNG" alt="Different Scenarios for Training" width="830" height="235">
<img src="./figures/SCL.PNG" alt="Different Scenarios for Training" width="830" height="235">
<img src="./figures/SL.PNG" alt="Different Scenarios for Training" width="830" height="235">

###### Top: CKA between the individual layers of networks trained by CL, SCL, and SL schemes through different scenarios. Bottom: Linear probe accuracy for each layer after residual connections. By comparing the CKA matrices through standard training (the first row) with the ones through adversarial training, we can see the emergence of more block structures in adversarial training scenarios, which means having more similar representations between layers. The results on the second row show the block structure is highly reduced in CL scheme after full fine-tuning or in AT-full AT scenario. Linear probe accuracies for learned representations also prove this finding.
