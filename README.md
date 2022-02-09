# TrilingualLossProj

\*\* *This repository holds the code used as the basis of Chris Crabtree's Masters thesis. 
 The thesis will be published [here](https://repositories.lib.utexas.edu/handle/2152/11) in the coming months, 
 but can also be viewed in the ```thesis/``` directory of this repository.* \*\*

This project attempts to optimize the alignment of vector representations of speech and images.
Improvement in this alignment should not only lead directly to increased performance down-stream tasks such as neural image retrieval, but also promises to aid in ongoing research efforts to utilize multi-modal data (e.g. vision and speech) to enhance tasks such as speech translation and low-resource ASR.
 

## Goal
We assess vector alignment by examining performance on *retrieval tasks*.
In these tasks, an input signal is given (e.g. a speech utterance describing an image) and the goal is to retrieve the correct candidate (e.g. the corresponding image) from a collection (e.g. a dataset of images).
To do this, we encode the input (either a speech utterance or an image) and the target collection using a neural encoders, then select the candidate from the collection that is the 'closest' according to a specified similarity measure.

Additionally, we seek to find an language agnostic vector representation space.
Specifically, if two utterances describe the same image, they should ideally have very similar vector representations and thus retrieve the same image.
In this project, we use a dataset consisting of nearly 100,000 images, each with utterances from three distant languages (*English, Japanese,* and *Hindi*) describing the image.
The vast differences between theses languages presents a difficult task, even for modern metric learning techniques.

Our goal, then, is to optimize the representation alignment of all languages and images over several state-of-the-art contrastive loss functions and training frameworks.

## Experiments
We explore four different aspects of vector alignment learning and their effect on overall performance: the loss function used, pooling mechanism, the loss complexity (as a function of the number terms used in the loss function), and the number of parameters used. 
__The details of each of our experimental designs can be found in the thesis linked to above.__

### Loss function Optimization
We explore five different contrastive loss functions (abbreviations used in results tables in parenthesis): 

1. Hypersphere loss (Hyper)
2. Triplet loss (Triplet)
3. InfoNCE (InfoNCE)
4. Masked Margin Softmax, Margin=1 (MMS)
5. Masked Margin Softmax, scheduled margin updates (MMS_Sch.)
6. Adaptive Masked Margin Softmax (MMS_Adp.)

Previous research in this area commonly used the triplet loss, but it has remained unclear which of the many contrastive losses that exist are best suited for this task.
Loss functions that required tuning (such as the hypersphere loss) were done so in a manner described in the thesis.

For each loss function we use separate CNN encoders for each language and the ResNet 50 CNN encoder for the images.
We used **recall at 1, recall at 5,** and **recall at 10** as our evaluation metrics.

Results of our experiments:

| Image Ret.   | E&I.R1       | H&I.R1       | J&I.R1       |     | E&I.R5       | H&I.R5       | J&I.R5       |     | E&I.R10     | H&I.R10      | J&I.R10     | Ep           |
| ------------ | ------------ | ------------ | ------------ | --- | ------------ | ------------ | ------------ | --- | ----------- | ------------ | ----------  | ------------ |
| Hyper        | 8.30\%       | 7.85\%       | 9.00\%       |     | 25.20\%      | 23.10\%      | 30.55\%      |     | 36.95\%     | 33.75\%      | 43.65\%     | 16           |
| Triplet      | 16.40\%      | 12.95\%      | 22.70\%      |     | 39.20\%      | 32.65\%      | 53.15\%      |     | 51.55\%     | 43.55\%      | 67.05\%     | 36           |
| InfoNCE      | 18.05\%      | 14.70\%      | 28.00\%      |     | 41.55\%      | 36.10\%      | 59.20\%      |     | 54.55\%     | 45.85\%      | 72.35\%     | 26           |
| MMS          | 12.65\%      | 16.15\%      | 25.50\%      |     | 36.65\%      | 36.45\%      | 57.45\%      |     | 50.30\%     | **47.60\%**  | 71.50\%     | 19           |
| MMS\_Sch.    | **18.85\%**  | **16.50\%**  | **28.35\%**  |     | 42.15\%      | **37.65\%**  | 58.05\%      |     | 54.90\%     | 46.85\%      | 71.95\%     | 24           |
| MMS\_Adp.    | 16.90\%      | 14.45\%      | 27.30\%      |     | **45.25\%**  | 36.90\%      | **61.05\%**  |     | **57.50\%** | 47.15\%      | **74.40\%** | 29           |


|Cross-Ling |        H\&E.R1 |        J\&E.R1 |        J\&H.R1 |        H\&E.R5 |        J\&E.R5 |        J\&H.R5 |       H\&E.R10 |       J\&E.R10 |       J\&H.R10 | Ep |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ----------   | ------------ | --------- |
|Hyper     |          5.90\% |          6.90\% |          6.40\% |         18.00\% |         21.30\% |         19.30\% |         27.05\% |         33.95\% |         28.95\% |         16 |
|Triplet   |          9.65\% |          8.05\% |          6.30\% |         23.35\% |         24.70\% |         21.80\% |         32.85\% |         35.85\% |         30.70\% |         36 |
|InfoNCE   |         11.55\% |         11.40\% | **10.50\%** |         27.25\% |         29.00\% | **27.30\%** |         37.55\% |         41.30\% | **36.80\%** |         26 |
|MMS      |          8.35\% |          7.90\% |          8.85\% |         22.10\% |         25.00\% |         24.70\% |         31.80\% |         36.85\% |         35.60\% |         19 |
|MMS\_Sch. | **12.00\%** |         11.25\% |          9.40\% | **28.45\%** | **32.25\%** |         26.35\% | **38.00\%** | **44.60\%** |         36.60\% |         24 |
|MMS\_Adp. |         10.65\% | **12.25\%** |         10.15\% |         26.30\% |         32.00\% |         25.40\% |         36.00\% |         42.40\% |         35.70\% |         29 |

Our results show that, at least for this dataset, the scheduled and adaptive MMS losses consistently outperform other losses.
Notably, these best performers represent a substantial improvement over the Triplet losses in common use previously.

### Remaining Experiments
Details of the remaining experiments and discussions of the results can be found in the thesis above

## Reproducibility 
We found there to be only mild variation on successive training runs. However the training scripts used to reproduce the results above can be found in ```./scipts```.
Results tables were created using ```./results/curate_results.py```.

