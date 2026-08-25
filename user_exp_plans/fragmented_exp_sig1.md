uniform head exp:
in mudularized_layer_probe_260813_01.md, each path has its own trained head. We would like to see if they have different performance on uniform head. 

do the following grid:
take k paths. k grid:[1,10,50,100,200,1000];
taking path strategy grid: 1. val-acc top-k paths; 2. random-k heads;(k=1 take canonical sequence); 3. unleaky: take random paths from top-100-4500(exclude top 100 paths)
use acc weighted train loss. train a uniform head for k-paths. 
show result, and for top-10 paths, show uniform-specific head gap; for top-100 paths, show mean and std of uniform-specific head gap;

other settings keep the same with mudularized_layer_probe_260813_01.md. 