We suspect whether the fail in fragmented_exp_sig1 is due to the geometry mismatch of different path hidden features. So, for top-10 paths, we each train a tiny translator in front canonical classification head. 
The translator is in form of residual low rank:$T_P(h_P)=h_P+B_PA_Ph_P,$ where $A_P\in R^{r\times d}, B_P\in R^{d\times r}.$ 
When r=4, this translator involves 768*8 trainable parameters, 4/75 compared with classification head. 
For grid, we try r=2,4,8,16. To see whether a translator could pull the hidden state of good paths back to the form that canonical classification head could readout. 
(260825)

Append exp(260826):
1. for top-10 paths, use their own W_p, do $\Delta W=W_p-W_c$, and svd analysis on $\Delta W$. 
2. this time, add bais to translator, and grid r=2,4,8,16,32,64,128;
3. replace hidden state similarity target with with with min ||(T(X_p)-T(X_c))W_c||_F^2