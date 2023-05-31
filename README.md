# CL-Robustness
Our goal is to understand whether there are differences in how contrastive learning learns the representation from data compared to supervised learning from the adversarial perspective. To this end, we conduct extensive experiments to evaluate the robustness of the following learning schemes:

- Contrastive Learning (CL): Within the standard framework of SimCLR, contrastive learning trains a base encoder by minimizing a contrastive loss over the representations projected into a latent space (Figure 1(a)). The extracted features will be employed to train a linear classifier on a downstream task as shown in Figure 1(a).
- Supervised Contrastive Learning (SCL): A supervised extension of contrastive learning  to avoid false positive pairs selection by leveraging the label information.
- Supervised Learning (SL): The network consists of a base encoder followed by a fully connected layer as a linear classifier, as shown in figure 1(b). In this case, cross entropy between the true and predicted labels is utilized for training the network parameters.

The training process in contrastive and supervised contrastive learning includes the following two phases:

-Pretraining Phase: The goal of this phase is to train the base encoder parameters $\bm{\theta}_{b}$ by minimizing a self-supervised loss $\mathscr{L}_{p}(\bm{\theta}_{b},\bm{\theta}_{ph})$ over a given dataset $\mathscr{D}_{p}$. Here $\bm{\theta}_{ph}$ is the parameters vector of the projection head used to map the base encoder output into a low dimensional latent space where the $\mathscr{L}_{p}$ is applied.

-Supervised Fine-tuning Phase: 
The goal of this phase is to train the linear classifier parameters $\bm{\theta}_{c}$ by minimizing the supervised loss $\mathscr{L}_f(\bm{\theta}_{c})$ over a labeled dataset $\mathscr{D}_f$.
The linear classifier learns to map the representations extracted during the pretraining phase to the labeled space, where $\mathscr{L}_f$ is the cross-entropy loss.

We examine the standard and robust training variations of the aforementioned training phases to compare the adversarial robustness across different learning schemes. Table 1 summarises all the studied training combinations for different possible scenarios of training phases in contrastive and supervised contrastive learning schemes.



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
