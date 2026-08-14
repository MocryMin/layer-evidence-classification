Result of gr2 is interesting and guiding. Before entering futher trail experiment, first do an analysis on current collected data. 

1. class-conditioned transition utility
for every (i,j), calculate: 
$$Gain_{(i,j)|c}=acc(i,j|y=c)-acc(i|y=c)$$
then calculate var_c(Gain).

2. path feature regression
for every path in gr2 task 4, calculate: len,repeat count, backward-jmp count, canonical-adjacent edge count, longest canonical run, start with [1]?, start with [1,2]?, distinct-layer count, disctinct-layer count/len,repeat_ratio = repeat_count / (len-1),backward_jump_ratio = backward_jump_count / (len-1),canonical_edge_ratio = canonical_adjacent_edge_count / (len-1),longest_canonical_run_ratio = longest_canonical_run / len,start_layer, tail_layer.
do regression test probes(linear, random forest...):
a. len, start_layer, tail_layer, on acc,linear
b. all features, on acc^{res},linear
c. b, but nonlinear(random forest regressor)
see: what property is explaining performance. 
where $\mu_k=E(acc(P)||P|=k),acc^{res}(p)=acc(P)-\mu_{|P|}$
remove deduplicate exact path. 

3. collect vocabulary
rank path performance from large 2 small. 
weighted count bigrams (i,j)s and trigram (i,j,k)s.
weight= acc^{res}/\sigma_{|P|}/(len-1) if bigram else (len-2)