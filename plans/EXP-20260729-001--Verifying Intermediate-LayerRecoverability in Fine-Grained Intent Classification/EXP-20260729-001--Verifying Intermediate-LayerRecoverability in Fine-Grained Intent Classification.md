## 1. BASIC INFO
- **Date**: 2026.7.29
- **State**: PLANNED
- **Git Commit**: 
- **MLflow run ID**: 
- **Artifact Path**: 
- **Data**: 
- **Model**: 
- **Seed**: 

## 2. PROBLEMS

Previous works have provided evidence that intermediate layers of LM possess certain evidence that the final layer might lack[arXiv:1810.07052](https://arxiv.org/abs/1810.07052)[arXiv:2006.04152](https://arxiv.org/abs/2006.04152)[Linguistic Knowledge and Transferability of Contextual Representations](https://aclanthology.org/N19-1112/).  
To verify whether mid-layer representations can really be used, and to what extent they can be exploited, we designed this experiment to verify 3 progressively fine-grained hypotheses which have been directly or indirectly produced by previous work on different downstream tasks. We verify these hypotheses in text-classification context with `DeBERTa-v3-base` on `CLINC150`: 
$H_1$ : An intermediate layer alone can show close performance on classification tasks to final layer. [arXiv:2006.04152](https://arxiv.org/abs/2006.04152)  
$H_1'$ : There exists some intermediate layers that can perform better on classification task than final layer. [arXiv:2502.02013](https://arxiv.org/abs/2502.02013)[arXiv:2412.13435](https://arxiv.org/abs/2412.13435)  
$H_2$ : Mid layer possesses certain `Recoverability` to final layer. [arXiv:2607.10391](https://arxiv.org/abs/2607.10391)  
DeeBERT[Xin et al., 2020], which used the early layers' representation to accelerate model inference, show that mid layers possess the potential to perform closely to a throughout forward, indirectly supported $H_1$; [Sawtell, et al., 2024], trained small linear classifiers on each layer, choosing the best according to different tasks;  [Skean O, et al., 2025], applied hidden representations from every layer to different downstream tasks involving clustering, text classification, found that in almost every single type of task, there exists one mid layer whose performance is much better than the final layer, both providing direct support for $H_2$. Some earlier work also identified destructive overthinking, where an intermediate prediction is correct but becomes incorrect with the piping-up of layers(Liu et al., 2019).  
A most recent work on Vertical Fusion [Di Salvo et al., 2026.7] formalizes this complementary relationship under the notion of **recoverability**, defined as the capacity of intermediate-layer probes to correctly classify samples misclassified by the final-layer probe. Specificially, they calculated the layer recoverability on ViT problems:  
$$
R_l=P\left(\hat{y}_l=y\mid \hat{y}_L\neq y \right)\tag1
$$
where $L$ is the count of model Transformer layers, and $1<l<L$.  
Building on this formulation, our experiment further studies recoverability in fine-grained text classification decomposed at class level. We define the **class-wise recoverability** of layer $l$ and class $c$ as:
$$
R_{l,c}=P(\hat y_l=y\mid y=c,\hat y_L\neq c)\tag2
$$
to gain deeper insight into how each mid layer performs over final layer on different classes.  

The main purpose of this experiment is to verify and reproduce $H_{1-2}$ on text classification context and to gather performance statistics including $R_{l,c}$ to determine whether the potential of mid layer compensation supports our future research.  

## 3. Experiment Steps

As a pilot verification experiment, we choose`DeBERTa-v3-base`as our major model, and `Qwen3-Embedding-0.6B`as side verification. The data set is `CLINC150`. If $H_{1-2}$ stand, we will use `modernBERT-base`and`modernBERT-large` as our future major targets. Specificially, we take $h_l(x) = H_l(x)[:,0,:]$, which means the`CLS`token.  

### 3.1 Baseline

For our baseline, we will train the output head $W_E^L\in R^{d\times C}$ while freeze the model backbone. For`DeBERTa-v3-base`and`CLINC150`, the hidden $d=768$, and class number $C=150$. So $W_E^L$ contains some`115k`trainable parameters.  

### 3.2 Training mid layers

This step contains 11 sub-steps, in each we independently train a different mid layer ranging from 1 to 11 of `DeBERTa-v3-base`which contains 12 transformer layers. For each layer, we freeze the model parameters, training 11 mid layer projection matrixes $W_E^l$, to get the layer-wise performance statistics. We train each $W_E^l$ separately because our goal is to fairly evaluate the performance of different layers, according to [Sawtell, et al., 2024].  

## 4. Collected Metrics

Although the full results of every single forward are stored locally, we explicitly collect or calculate the following statistics.  

| Name             | Step    | Inf                                        | Use                          |
| ---------------- | ------- | ------------------------------------------ | ---------------------------- |
| Acc              | 3.1;3.2 | test accuracy                              | verify $H_1,H_1'$            |
| macro-F1         | 3.1     | -                                          | future use                   |
| NLL              | 3.1;3.2 | Negative Log-Likelihood                    | future use                   |
| margin           | 3.1;3.2 | Top1-Top2                                  | future use                   |
| entropy          | 3.1;3.2 | -                                          | future use                   |
| confusion matrix | 3.1;3.2 | -                                          | future use                   |
| ECE              | 3.2     | M=10                                       |                              |
| $R_l$            | -       | recoverability                             | verify $H_2$                 |
| $H_l$            | -       | harm rate(oppose to recoverability)        | verify $H_2$                 |
| $R_{l,c}$        | -       | class-wise recoverability                  | verify $H_2$ with fine grain |
| $R_{oracle}$     | -       | Theoretical R with `Oracle layer choice`   | upper bound                  |
| $Acc_{oracle}$   | -       | Theoretical Acc with `Oracle layer choice` | upper bound                  |
| $H_{l,c}$        | -       | class-wise harm rate                       | verify $H_2$ with fine grain |

For some unspecified metrics above, we provide clear mathematical definition for any third party who wants to reproduce the same experiment outcome:  
$$
NLL_l(x)=-log(p_y^l(x))
$$
$$
NLL_l=\dfrac{1}{n}\sum_{i=1}^nNLL_l(x_i)
$$

$$
margin_l(x)=p_{top1}^l(x)-p_{top2}^l(x)
$$
$$
margin_l=\dfrac{1}{n}\sum_{i=1}^nmargin_l(x)
$$

$$
entropy_l(x)=-\sum_{i=1}^Cp_i^l(x)log(p_i^l(x))
$$
For ECE, we take the bucket size as $10$, and $I_m$ for samples falling to $m$th bucket. We also have: 
$$
conf_m=\dfrac{1}{|I_m|}\sum_{i\in I_m}\hat p_i,\quad acc_m=\dfrac{1}{|I_m|}\sum_{i\in I_m}1(\hat y_i=y_i)
$$
then we have ECE: 
$$
ECE=\sum_{m=1}^M\dfrac{|I_m|}{n}|acc_m-conf_m|
$$
$R_l$ and $R_{l,c}$ has been provided above, here we provide $H_l$: 
$$
H_l=P(\hat y_l\neq y\mid \hat y_L=y)
$$
which means when final layer is right, how often is layer $l$ wrong. And accordingly class-wise $H_{l,c}$: 
$$
H_{l,c}=P(\hat y_l\neq y\mid y=c,\hat y_L=y)
$$

> Note that `CLINC150` only has 4500 test samples(30 per class), so class-wise recoverability could potentially be faced with cases where there are no enough error samples for a certain class. So when reporting $R_{l,c}$ and $H_{l,c}$ , we report the unsimplified real-number fraction like $\frac{3}{6}$ instead of decimal ratio like 0.5.  

We compute the theoretical upper bound under the assumption of an `Oracle layer chooser` who can always choose the very layer that tells the right answer given sample $x$[Di Salvo et al., 2026.7]: 
$$\operatorname{Acc}_{\mathrm{oracle}} = P\left( \hat y_L=y \ \lor\ \exists l:\hat y_l=y \right)
$$and correspondingly the: 
$$
R_{oracle}=P(\exists l<L:\hat y_l=y\mid \hat y_L\neq y)
$$

## 5. Hypotheses Judgement

With the metrics above, we are able to judge whether $H_{1-2}$ stand. 
- For $H_1$ , we define a non-inferiority tolerance $\epsilon$ , if $\exists l<L$, so that$\Delta A_1=Acc_L-Acc_l<\epsilon_1$ , then we accept $H_1$ . Here, we take $\epsilon=0.02$.  
- For $H_1'$ , if $\exists l<L$ and $Acc_l>Acc_L$，
- For $H_3$ , we accept $H_3$ only if all the following are satisfied: 
	1. **Remarkable Improving Potential**: $\Delta A_2=Acc_{oracle}-Acc_{L}>\epsilon_2$ . Here we take $\epsilon_2=0.05$.  
	2. **No occasionality**: 
		**a**. Recoverable sample scale should be prominant, this is indirectly conveyed by defining $\epsilon_2=0.05$, which means there are at least 225 potentially recoverable samples.  
		**b. Class-wise distribution of recoverability.** Recoverability should not be disproportionately concentrated in only a small subset of the classes on which the final layer makes errors. To quantify this property, we compare the class distribution of all final-layer errors with the class distribution of oracle-recoverable errors.
		For seed $s$ and class $c$, let $n^{\mathrm{err}}_{s,c}$ denote the number of samples from class $c$ misclassified by the final layer, and let $n^{\mathrm{rec}}_{s,c}$ denote the number of such samples that are correctly classified by at least one intermediate layer. We define:$$
e_{s,c}
=
\frac{n^{\mathrm{err}}_{s,c}}
{\sum_j n^{\mathrm{err}}_{s,j}},
\qquad
r_{s,c}
=
\frac{n^{\mathrm{rec}}_{s,c}}
{\sum_j n^{\mathrm{rec}}_{s,j}}.
$$Here, $e_s$ represents the class distribution of all final-layer errors, whereas $r_s$ represents the class distribution of oracle-recoverable errors. We then define their midpoint distribution as:$$
m_{s,c}
=
\frac{e_{s,c}+r_{s,c}}{2}.
$$The normalized class-wise Jensen--Shannon divergence is:$$
D_{\mathrm{JS},s}^{\mathrm{class}}
=
\frac{
\frac{1}{2}D_{\mathrm{KL}}(e_s \Vert m_s)
+
\frac{1}{2}D_{\mathrm{KL}}(r_s \Vert m_s)
}{
\log 2
}
\in [0,1].
$$A value close to $0$ indicates that recoverable errors are distributed across classes approximately in proportion to the final layer's original error distribution. A larger value indicates that recoverability is disproportionately concentrated in, or absent from, particular classes.

For all $H_{1-2}$, we require cross seed confidence to make sure the results stands across different seeds. Specificially, we sample 10 different seeds and require that: 
$$\epsilon_1\geq CI_{95\%}(\hat{\Delta A_1^{up}})$$$$\epsilon_2\in CI_{95\%}[\hat {\Delta A_2^{low}},\hat{\Delta A_2^{up}}]$$