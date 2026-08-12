In this fragmented experiment group, you need to conduct the following individual experiments. 

## 1. qwen3embed0p6bEXP1ver
qwen3-embedding-0.6b is in our resources. It is a good target for sideway verify EXP1 and EXP3's results. 
do the following(single seed):
1. qwen3-embedding-0.6b last token variance. 
2. qwen3-embedding-0.6b \times {plain(if no collapse in 1), LN plain, centered plain(if collapsed in 1), ridge(grided)}. AdamW:full batch,wd=1e-2 if not collapse else wd=0(adam); smoke lr; 1w ep, early stop at val acc. show layer acc, recoverability. 

## 2. modernBERT-baseEXP1ver
same as ## 1, but use CLS. 

## 3. WOS-46985features
count basic features and statistics like sample class distribution on train/val/test, sample len distribution and so on dataset: WOS-46985. 

## 4. DeBERTa-v3-baseWOS-46985Baseline
take WOS-46985 as plain classification(134 L2). verify:
1. cls collapse?
2. test: {plain(if not collapse), LN plain, centered, ridge(grided)}. adamW: full batch,wd=1e-2 if not collapse else wd=0(adam);smoke lr; 1w ep, early stop at val acc. 
3. collect baseline acc only. 

## 5. ModernBERT-baseWOS-46985Baseline
as 4. 