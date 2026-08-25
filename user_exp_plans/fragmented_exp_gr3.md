tonight's experiment plan: 

1. For ./agent_BuildReports/fragmented-experiments/DeBERTaV3BaseWOS46985Baseline_260812_04.md, add a. layer recoverability(def in EXP-001). b. class-wise recoverability analysis. (since in WOS46985 there are enough error samples). Same on ModernBERTBase-WOS46985. both use ridge result. Write report in class-wiseRecoverabilityWOS46985_260818_01.md

2. Conduct the same data analyses(without step4 random path analysis) in ./user_exp_plans/gr2_data_analysis_plan.md step 1, but on ./agent_BuildReports/fragmented-experiments/DeBERTaV3BaseWOS46985LayerProbe_260814_01.md. Write Analysis_DeBERTaV3BaseWOS46985LayerProbe_260818_02.md.  

3. fine-tuning baseline on DeBERTaV3Base-WOS46985, optimize on final layer performance with LN-plain classifier, full/attention-only(do both), AdamW, wd=0.01, bs=32, microbs=8 grad accumul, 5ep, 4 val acc checkpoints/ep(train an init probe first at frozen backbone; then use this trained probe in fine-tuning). report on best ckp: a. whether variance compression happen on fine-tuned(def: EXP-001); b. all layer probe acc on fine-tuned(LN, ridge). smoke to optimize fine-tune config. Write FT-BaselineDeBERTaV3BaseWOS46985_260818_03.md. 


