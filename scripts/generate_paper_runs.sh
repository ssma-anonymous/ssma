########## enzymes ##########
# gat #
python train.py -cn enzymes layer_type=gat aggr=SSMA max_neighbors=6 mlp_compression=0.1 runs=5 use_attention=true
# gat2 #
python train.py -cn enzymes layer_type=gat2 aggr=SSMA max_neighbors=3 mlp_compression=1 runs=5 use_attention=true
# gcn #
python train.py -cn enzymes layer_type=gcn aggr=SSMA max_neighbors=2 mlp_compression=0.25 runs=5 use_attention=true
# gin #
python train.py -cn enzymes layer_type=gin aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
# graphgps #
python train.py -cn enzymes layer_type=graphgps aggr=SSMA max_neighbors=5 mlp_compression=1 runs=5 use_attention=false
# pna #
python train.py -cn enzymes layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=6 mlp_compression=0.1 runs=5 use_attention=true
########## imdb_binary ##########
# gat #
python train.py -cn imdb_binary layer_type=gat aggr=SSMA max_neighbors=3 mlp_compression=0.1 runs=5 use_attention=false
# gat2 #
python train.py -cn imdb_binary layer_type=gat2 aggr=SSMA max_neighbors=3 mlp_compression=0.25 runs=5 use_attention=false
# gcn #
python train.py -cn imdb_binary layer_type=gcn aggr=SSMA max_neighbors=4 mlp_compression=0.75 runs=5 use_attention=false
# gin #
python train.py -cn imdb_binary layer_type=gin aggr=SSMA max_neighbors=6 mlp_compression=0.5 runs=5 use_attention=false
# graphgps #
python train.py -cn imdb_binary layer_type=graphgps aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=false
# pna #
python train.py -cn imdb_binary layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=4 mlp_compression=0.5 runs=5 use_attention=false
########## mutag ##########
# gat #
python train.py -cn mutag layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=0.5 runs=5 use_attention=false
# gat2 #
python train.py -cn mutag layer_type=gat2 aggr=SSMA max_neighbors=3 mlp_compression=0.25 runs=5 use_attention=false
# gcn #
python train.py -cn mutag layer_type=gcn aggr=SSMA max_neighbors=3 mlp_compression=0.25 runs=5 use_attention=true
# gin #
python train.py -cn mutag layer_type=gin aggr=SSMA max_neighbors=4 mlp_compression=0.5 runs=5 use_attention=false
# graphgps #
python train.py -cn mutag layer_type=graphgps aggr=SSMA max_neighbors=4 mlp_compression=0.5 runs=5 use_attention=false
# pna #
python train.py -cn mutag layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
########## ogbg_molhiv ##########
# gat #
python train.py -cn ogbg_molhiv layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=0.75 runs=5 use_attention=false
# gat2 #
python train.py -cn ogbg_molhiv layer_type=gat2 aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=false
# gcn #
python train.py -cn ogbg_molhiv layer_type=gcn aggr=SSMA max_neighbors=4 mlp_compression=0.5 runs=5 use_attention=true
# gin #
python train.py -cn ogbg_molhiv layer_type=gin aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
# graphgps #
python train.py -cn ogbg_molhiv layer_type=graphgps aggr=SSMA max_neighbors=4 mlp_compression=1 runs=5 use_attention=true
# pna #
python train.py -cn ogbg_molhiv layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=4 mlp_compression=0.25 runs=5 use_attention=false
########## ogbg_molpcba ##########
# gat #
python train.py -cn ogbg_molpcba layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=false
# gat2 #
python train.py -cn ogbg_molpcba layer_type=gat2 aggr=SSMA max_neighbors=4 mlp_compression=1 runs=5 use_attention=false
# gcn #
python train.py -cn ogbg_molpcba layer_type=gcn aggr=SSMA max_neighbors=3 mlp_compression=1 runs=5 use_attention=false
# gin #
python train.py -cn ogbg_molpcba layer_type=gin aggr=SSMA max_neighbors=4 mlp_compression=1 runs=5 use_attention=true
# graphgps #
python train.py -cn ogbg_molpcba layer_type=graphgps aggr=SSMA max_neighbors=3 mlp_compression=1 runs=5 use_attention=false
# pna #
python train.py -cn ogbg_molpcba layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=2 mlp_compression=0.1 runs=5 use_attention=true
########## ogbn_arxiv ##########
# gat #
python train.py -cn ogbn_arxiv layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=0.1 runs=5 use_attention=false
# gat2 #
python train.py -cn ogbn_arxiv layer_type=gat2 aggr=SSMA max_neighbors=3 mlp_compression=0.5 runs=5 use_attention=false
# gcn #
python train.py -cn ogbn_arxiv layer_type=gcn aggr=SSMA max_neighbors=2 mlp_compression=0.5 runs=5 use_attention=true
# gin #
python train.py -cn ogbn_arxiv layer_type=gin aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
# graphgps #
python train.py -cn ogbn_arxiv layer_type=graphgps aggr=SSMA max_neighbors=5 mlp_compression=0.25 runs=5 use_attention=true
# pna #
python train.py -cn ogbn_arxiv layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
########## ogbn_products ##########
# gat #
python train.py -cn ogbn_products layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
# gat2 #
python train.py -cn ogbn_products layer_type=gat2 aggr=SSMA max_neighbors=3 mlp_compression=0.5 runs=5 use_attention=true
# gcn #
python train.py -cn ogbn_products layer_type=gcn aggr=SSMA max_neighbors=2 mlp_compression=0.25 runs=5 use_attention=true
# gin #
python train.py -cn ogbn_products layer_type=gin aggr=SSMA max_neighbors=6 mlp_compression=0.1 runs=5 use_attention=true
# graphgps #
python train.py -cn ogbn_products layer_type=graphgps aggr=SSMA max_neighbors=4 mlp_compression=0.75 runs=5 use_attention=true
# pna #
python train.py -cn ogbn_products layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=4 mlp_compression=1 runs=5 use_attention=true
########## peptides_func ##########
# gat #
python train.py -cn peptides_func layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=0.5 runs=5 use_attention=false
# gat2 #
python train.py -cn peptides_func layer_type=gat2 aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
# gcn #
python train.py -cn peptides_func layer_type=gcn aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
# gin #
python train.py -cn peptides_func layer_type=gin aggr=SSMA max_neighbors=2 mlp_compression=0.1 runs=5 use_attention=true
# graphgps #
python train.py -cn peptides_func layer_type=graphgps aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=false
# pna #
python train.py -cn peptides_func layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=2 mlp_compression=0.5 runs=5 use_attention=true
########## peptides_struct ##########
# gat #
python train.py -cn peptides_struct layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=0.1 runs=5 use_attention=false
# gat2 #
python train.py -cn peptides_struct layer_type=gat2 aggr=SSMA max_neighbors=2 mlp_compression=0.75 runs=5 use_attention=false
# gcn #
python train.py -cn peptides_struct layer_type=gcn aggr=SSMA max_neighbors=3 mlp_compression=1 runs=5 use_attention=false
# gin #
python train.py -cn peptides_struct layer_type=gin aggr=SSMA max_neighbors=3 mlp_compression=0.75 runs=5 use_attention=false
# graphgps #
python train.py -cn peptides_struct layer_type=graphgps aggr=SSMA max_neighbors=2 mlp_compression=0.25 runs=5 use_attention=false
# pna #
python train.py -cn peptides_struct layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
########## proteins ##########
# gat #
python train.py -cn proteins layer_type=gat aggr=SSMA max_neighbors=7 mlp_compression=0.75 runs=5 use_attention=true
# gat2 #
python train.py -cn proteins layer_type=gat2 aggr=SSMA max_neighbors=4 mlp_compression=0.1 runs=5 use_attention=false
# gcn #
python train.py -cn proteins layer_type=gcn aggr=SSMA max_neighbors=5 mlp_compression=0.75 runs=5 use_attention=true
# gin #
python train.py -cn proteins layer_type=gin aggr=SSMA max_neighbors=3 mlp_compression=0.25 runs=5 use_attention=true
# graphgps #
python train.py -cn proteins layer_type=graphgps aggr=SSMA max_neighbors=4 mlp_compression=0.75 runs=5 use_attention=false
# pna #
python train.py -cn proteins layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=7 mlp_compression=1 runs=5 use_attention=false
########## ptc_mr ##########
# gat #
python train.py -cn ptc_mr layer_type=gat aggr=SSMA max_neighbors=4 mlp_compression=0.5 runs=5 use_attention=false
# gat2 #
python train.py -cn ptc_mr layer_type=gat2 aggr=SSMA max_neighbors=3 mlp_compression=0.5 runs=5 use_attention=true
# gcn #
python train.py -cn ptc_mr layer_type=gcn aggr=SSMA max_neighbors=4 mlp_compression=0.5 runs=5 use_attention=true
# gin #
python train.py -cn ptc_mr layer_type=gin aggr=SSMA max_neighbors=2 mlp_compression=0.1 runs=5 use_attention=false
# graphgps #
python train.py -cn ptc_mr layer_type=graphgps aggr=SSMA max_neighbors=2 mlp_compression=0.5 runs=5 use_attention=true
# pna #
python train.py -cn ptc_mr layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=2 mlp_compression=0.25 runs=5 use_attention=true
########## zinc ##########
# gat #
python train.py -cn zinc layer_type=gat aggr=SSMA max_neighbors=2 mlp_compression=0.75 runs=5 use_attention=true
# gat2 #
python train.py -cn zinc layer_type=gat2 aggr=SSMA max_neighbors=2 mlp_compression=1 runs=5 use_attention=true
# gcn #
python train.py -cn zinc layer_type=gcn aggr=SSMA max_neighbors=2 mlp_compression=0.5 parameter_budget=400000 runs=5 use_attention=true
# gin #
python train.py -cn zinc layer_type=gin aggr=SSMA max_neighbors=2 mlp_compression=0.5 parameter_budget=400000 runs=5 use_attention=false
# graphgps #
python train.py -cn zinc layer_type=graphgps aggr=SSMA max_neighbors=3 mlp_compression=0.5 runs=5 use_attention=false
# pna #
python train.py -cn zinc layer_type=pna aggr=[mean,max,min,std,SSMA] max_neighbors=4 mlp_compression=0.75 parameter_budget=800000 runs=5 use_attention=true
