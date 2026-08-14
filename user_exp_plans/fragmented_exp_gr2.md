This fragmented experiment is a probing experiment, gathering some information. 
Fragmented experiment name: mudularized_layer_probe

model: DeBERTa-v3-base, data: CLINC150.
ridgeClassifier,\alpha=1e-6, EXP-003 config.

1. take every layer as 1st layer and do a 1 layer only probing. demonstrate acc compared with their in-raw-place performance. 

2. greedy
take the highest acc layer in 1 at the head of queue, do greedy strategy:
greedy choose next layer to append: test all 12 layers one by one to tail of current layer queue, forward from head2tail, probe at tail hidden state. take the layer out of 12 which added the most acc at this step, append it to the tail. (allow repeat)
do a max queue len=50 converge test. take down acc at each len. note whether there exist negative gain steps and whether max acc exists after any negative step. 

3. do an 1-prefix add-layer gain:
save the result in 1. then do acc{i,j}-acc{i} for all i,j in 1-12. and get a 12*12 add-layer gain matrix. 
and also calculate a add-layer abs gain matrix: $S_{ij}=A_{ij}-\max(A_i,A_j).$

4. do a random sampled path collect. 
create $n$ random 3-12-step repeatable layer path,like [2,3,1,1,9,2,1,4,5,6,4,11], [12,3,10]... forward to last layer of each path and take down probing acc. you have several hours to run exp-gr2(exactly stop point: now is about 23pm, i'll go to sleep, you can train through night. stop at tomorrow 8 a.m., when i need to go to work and this laptop will be carried, without electricity souce), so smoke a good $n$. trick: you can reuse prefix cache, like first create [1,2,3], calculate cache, and then [1,2,3,8,3], reuse cache. use this trick properly but do not break randomness. 
advice: uniform 3-12 len. random generate first, then build a prefix trie, reuse cache. 
for every path, save
```
path
length
tail_layer
val_accuracy
macro_f1
predictions[3000]
```

for 1-4, use train to fit ridge, use val to get acc. 