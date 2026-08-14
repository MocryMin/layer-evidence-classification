## 1. BASIC INFO
- **Date**: `2026.8.10` (run 2026-08-11)
- **State**: COMPLETE
- **Verdict**: H1, H1', H2 all **very_strong** (all three probe families support)
- **Git Commit**: `58aee5e` (run on `9055c92` dirty)
- **MLflow run ID**: `364bf897628d443189b1ae6f1288fc6e`
- **Artifact Path**: `artifacts/EXP-20260810-003/`
- **Data**: `CLINC150`
- **Model**: `Deberta-v3-base`

## 2. Problem to Handle

In this experiment, we have enough information to process the undone verification in EXP-001 with $H_{1-2}$:  

$H_1$ **Non-inferiority**: An intermediate layer alone can show close performance on classification tasks to final layer. [arXiv:2006.04152](https://arxiv.org/abs/2006.04152)  

$H_1'$ **Superiority**: There exists some intermediate layers that can perform better on classification task than final layer. [arXiv:2502.02013](https://arxiv.org/abs/2502.02013)[arXiv:2412.13435](https://arxiv.org/abs/2412.13435)  

$H_2$ **Recoverability**: Mid layer possesses certain `Recoverability` to final layer. [arXiv:2607.10391](https://arxiv.org/abs/2607.10391)  

Specificially, direct support for $H_1$ and $H_1'$ has been spotted during EXP-002 where mid layers showed greater accuracy than final layer under equal configuration. But that's not enough for us to accept $H_1$ and $H_1'$ since the accept conditions step for $H_{1-2}$ in EXP-001 section 5 have not been tested. 
In this experiment we will use three probing methods judged in EXP-002 on 10 different seeds to gather enough statistics. Specificially, we use the following probing/optimizing method:  
1. **Centered Plain**: Centering plain possesses exactly the same parameter size as our initial plain probe and preserves the hypothesis class of a biased linear classifier while changing its optimisation geometry.
2. **LN Plian**: A robustness control to centered plain, introducing sample-dependent normalization before the linear classifier. It involves $2\times d_{model}$ additive parameters(+1.33pp) to plain probing method.
3. **RidgeClassifier(fix grided $\alpha$)**: Our best spotted method so far(OLS), keeping pure plain probe as it is. We search the $\alpha$ for different layers with in a fixed $\alpha$ grid as EXP-002(add $\alpha=0$ which equals OLS to EXP-002 grid; we choose $\alpha$ on val).  

## 3. Experiment Process

### 3.1 Experimental details
After EXP-001 and EXP-002, we have clear picture of what to do and what to collect. In this experiment, we will collect the metrics in EXP-001 section 4, use them to calculate test indicators in section 5 and finally give our assertion for $H_{1-2}$ . After that, we would decide what our next step is.  
Worth noticing is that pure RidgeClassifier provides **decision scores**, optimized from ridged least square. So even if we take softmaxed decision scores, it can not be compared equally with the other two methods where the softmaxed p takes part in optimizing process. **So, for RidgeClassifier, we only collect $Acc_l, R_l, H_l，R_{oracle},Acc_{oracle}$ which are needed for $H_{1-2}$ judgement**. Also, since RidgeClassifier is seed-agnostic, we only calculate it once. And we do not do cross-seed stability test on ridge.  
To prevent potential undertraining, we demand: 
```
{
	max_ep=1000,
	min_ep=100,
	patience=100,
	min_delta=1e-4,
	monitor = val acc
}
```
If a method did not stop within 1000 ep, we will tag it when reporting. Still we select checkpoints based on val acc.  
For centering and LN, we use `AdamW, lr=1e-2, full batch, wd=0, xavier` based on EXP-002 evidence that full-batch optimization reduces the undertraining risk while others keep the same as EXP-001's original configuration. 
For other details, we use frozen backbone of `DeBERTa-v3-base`; We reuse the hidden state cache in EXP-001; Every probe family compares intermediate layers against its own layer-12 baseline. Since scikit-learn requires that $\alpha\geq 0$, so the ridge grid $\alpha=0$ is realized by OLS with the same target encoding. 

### 3.2 Accept Protocol

Since there are three tested probing method. So we define our accept protocol for occasions where three methods show different results. 
We take centering as the main method for it is the closest to EXP-001 setting. We take LN plain as its robustness conditioning control and ridge linear as its solver-independent reference. 
If main accepts, we regard the hypothesis as acceptable. If all three accept, we regard the hypothesis as very strong. If main refuses but LN/ridge accepts, we assume that evidence for intermediate-layer information exists, but the performance ordering is probe-sensitive.  
If the primary protocol satisfies the pre-specified EXP-001 acceptance criterion, the hypothesis is considered supported.  