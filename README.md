# HGNNStudy — Reproduction of DHG-Bench Node Classification

This repository documents my reproduction of the **node classification** experiments in *DHG-Bench: A Comprehensive Benchmark for Deep Hypergraph Learning* (ICLR 2026). All **20 baselines** are evaluated on **13 hypergraph datasets**, using **5 random seeds** each and the per-dataset hyperparameters released with the benchmark.

---

## 1. Environment

All experiments were run on a single local workstation. The exact software stack is:

| Component | Version / Setting |
| --- | --- |
| Operating system | Windows 10 |
| GPU | NVIDIA RTX 2080 Ti (22 GB VRAM) |
| Python | 3.10.22 |
| PyTorch | 2.6.0+cu124 (CUDA 12.4) |
| PyTorch Geometric | 2.8.1 |
| NumPy | 1.24.3 |
| openpyxl | 3.1.5 |

The Python interpreter used throughout is the conda environment `dhgbench`:

```bash
D:\anaconda3\envs\dhgbench\python.exe
```

Key dependencies (per the official requirements): `torch`, `torch_geometric`, `torch-cluster`, `torch-scatter`, `torch-sparse`, `torch-spline-conv`, `deeprobust==0.2.11`, `numpy==1.24.3`, `dask==2024.8.0`.

---

## 2. Data Preparation

1. Download the benchmark archive `data.zip` from the Hugging Face dataset hub:

   https://huggingface.co/datasets/Philley/DHG-Bench-dataset

2. Extract it at the repository root so that a `data/` directory is created:

```bash
unzip data.zip
```

The resulting directory layout is:

```
data
+-- fair_data
+-- hete_data      (actor, amazon, pokec, twitch)
+-- hgcls_data
+-- large_data     (trivago)
+-- pyg_data       (converted PyG datasets, generated on first run)
+-- trad_data      (cora, pubmed, coauthor_cora, coauthor_dblp, NTU2012,
                   ModelNet40, walmart-trips, yelp, ...)
```

On the first run each dataset is converted into a PyG dataset and cached under `data/pyg_data/hypergraph_dataset_updated/`. The 13 node-classification datasets and their raw locations are:

| CLI dataset name | Raw location |
| --- | --- |
| `cora` | trad_data/cocitation/cora |
| `coauthor_cora` | trad_data/coauthorship/cora |
| `NTU2012` | trad_data/NTU2012 |
| `ModelNet40` | trad_data/ModelNet40 |
| `actor` | hete_data/actor |
| `twitch` | hete_data/twitch |
| `pubmed` | trad_data/cocitation/pubmed |
| `amazon` | hete_data/amazon |
| `coauthor_dblp` | trad_data/coauthorship/dblp |
| `walmart-trips-100` | trad_data/walmart-trips |
| `yelp` | trad_data/yelp |
| `pokec` | hete_data/pokec |
| `trivago` | large_data/trivago |

---

## 3. How to Run

All commands run from the `dhgbench/` subdirectory, where `main.py`, `lib_yamls/` and `results/` reside — not from the repository root. Enter it first:

```bash
cd dhgbench
```

A single node-classification run then has the following form:

```bash
python main.py --dname=<dataset> --task_type=node_cls --method=<method> --num_seeds=5
```

Both the YAML config (`./lib_yamls/...`) and the result folder (`./results`) use paths relative to this working directory. The datasets themselves remain in the repository-root `data/` folder and are reached through the `../data` path from inside `dhgbench/`, so the data must be prepared before entering the directory.

Parameter notes:

- `--dname`: dataset CLI name (see the table above).
- `--task_type`: always `node_cls` here.
- `--method`: one of the 20 baselines.
- `--num_seeds`: number of random seeds; we use 5.
- **`--is_default` is intentionally never passed.** With its parser default (`False`), the framework loads the *dataset-specific* tuning section of the YAML file, which is exactly the per-dataset configuration used in the paper. Passing `--is_default=True` would instead load the global `default` section and give different results.
- The benchmark split is 50%/25%/25% for train/validation/test, enforced by `--use_bench_prop=True` (the parser default).

The helper `run_all_node_cls.py` runs every dataset x method combination in order, shows live epoch output, skips combinations that fail, and records status in `run_node_cls_status.json`. Start it with `run_all_node_cls.bat`.

Result JSON files are written to `dhgbench/results/node_cls/<dataset>/<method>.json`, and the aggregated workbook is `node_cls_results.xlsx`.

### 3.1 Re-running a single problematic combination in PyCharm

When a specific dataset x method combination fails or gives an unexpected result, re-run it on its own in PyCharm instead of launching the whole batch. Each saved run configuration targets one combination and is named `<dataset>_<method>` (e.g. `actor_LEGCN`, `yelp_HyperGCN`, `amazon_HyperGT`).

1. Open the repository folder in PyCharm and set the project interpreter to the conda environment `dhgbench` (**Settings → Project → Python Interpreter**).
2. Go to **Run → Edit Configurations**, click **+ → Python**, and fill in:
   - **Name:** `<dataset>_<method>`
   - **Script path:** the `main.py` inside the `dhgbench/` folder
   - **Parameters:** `--dname=<dataset> --task_type=node_cls --method=<method> --num_seeds=5`
   - **Working directory:** the absolute path to the `dhgbench/` folder — this is mandatory, because `main.py` resolves `lib_yamls/` and `results/` relative to the working directory
   - **Python interpreter:** `dhgbench`
3. Do **not** add `--is_default`. Leaving it at the default `False` loads the dataset-specific YAML section, matching the paper.
4. Save the configuration, select it from the drop-down in the top-right corner, and press **Run** — or **Debug** to set breakpoints and inspect the failure step by step.
5. The run prints live epoch output and writes the result to `results/node_cls/<dataset>/<method>.json`. Repeat for every problematic combination: the batch helper skips failed cells, so these single runs are how the recoverable failures are re-tested.

---

## 4. Complete Commands and Hyperparameters by Dataset

For every dataset below, the first block lists the complete command for all 20 methods, and the second block lists the effective hyperparameters of each method as loaded from its YAML file. The paper's UniGNN row is reproduced with **UniGCNII** (paper Sec. 3.2). All commands below are run from the `dhgbench/` directory after `cd dhgbench` (see Section 3).

### 4.1 Cora (`cora`)

```bash
python main.py --dname=cora --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=cora --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.8, lr=0.001, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0, lr=0.001, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.001, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=150, dropout=0.5, lr=0.001, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0
LEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.1, lr=0.01, wd=0.01
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=128, epochs=50, dropout=0.5, lr=0.01, wd=1e-05, HyperND_ord=2, HyperND_tol=1e-05, HyperND_steps=100, restart_alpha=0.5, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=64, epochs=50, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0.5, prop_step=16, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=512, epochs=10, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0.01, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=2, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, MLP_num_layers=3, epochs=30, dropout=0.7, alpha=0.1, lr=0.001, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=2, MLP_hidden=256, epochs=50, lr=0.001, wd=0, dropout=0.8, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=0.3, lamda=0.8, reduce=mean
AllSetformer     [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, MLP_num_layers=2, epochs=100, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=64, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=200, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=1, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.2
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.005, wd=0.001, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=true, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=80, dropout=0.2, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=8, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=20, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=128, M=5, Mlst=[5], combine=concat
```

### 4.2 Coauthor-Cora (`coauthor_cora`)

```bash
python main.py --dname=coauthor_cora --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=coauthor_cora --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.8, lr=0.001, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.02, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0
HyperGCN         [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.8, lr=0.001, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0
LEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.1, lr=0.01, wd=0.01
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.001, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.3, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=64, epochs=50, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0.1, prop_step=16, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=4, embedding_hidden=128, MLP_hidden=256, epochs=15, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=1, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=1, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=1024, MLP_num_layers=3, epochs=60, dropout=0.7, alpha=0.3, lr=0.001, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=1e-05, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=2, MLP_hidden=256, epochs=50, lr=0.001, wd=0, dropout=0.8, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=0.3, lamda=0.8, reduce=mean
AllSetformer     [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, MLP_num_layers=2, epochs=300, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=500, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=1, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.1
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=256, epochs=100, dropout=0.5, lr=0.005, wd=0.001, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=0.001, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=8, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=256, M=5, Mlst=[5], combine=concat
```

### 4.3 NTU2012 (`NTU2012`)

```bash
python main.py --dname=NTU2012 --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=NTU2012 --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=400, dropout=0.8, lr=0.001, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=300, dropout=0, lr=0.001, wd=0, use_bn=true
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.5, lr=0.01, wd=0, heads=4, concat=false, use_bn=true
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.5, lr=0.001, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0
LEGCN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=300, dropout=0.5, lr=0.01, wd=0
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0.001, HyperND_ord=2, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.8, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.5, lr=0.01, wd=0, lam0=10, lam1=10, alpha=0.1, prop_step=8, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, epochs=100, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=300, dropout=0.5, lr=0.1, wd=0, activation=relu, neg_slope=0.01, gamma=0.01, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, epochs=200, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=2, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, MLP_num_layers=2, epochs=300, dropout=0.7, alpha=0.1, lr=0.001, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=256, epochs=400, dropout=0.8, lr=0.1, wd=0, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=1, MLP_hidden=512, epochs=200, lr=0.01, wd=0, dropout=0.5, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=0.3, lamda=0.1, reduce=mean
AllSetformer     [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, MLP_num_layers=2, epochs=100, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=64, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=1, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.2
HyperGT          [dataset section] All_num_layers=1, embedding_hidden=128, hidden_channels=512, epochs=200, dropout=0.5, lr=0.001, wd=0.01, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=500, dropout=0.2, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=8, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=50, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=128, M=5, Mlst=[5], combine=concat
```

### 4.4 ModelNet40 (`ModelNet40`)

```bash
python main.py --dname=ModelNet40 --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=ModelNet40 --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=400, dropout=0.8, lr=0.001, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=300, dropout=0, lr=0.001, wd=0, use_bn=true
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.01, wd=0, heads=4, concat=false, use_bn=true
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.5, lr=0.001, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0
LEGCN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=300, dropout=0.5, lr=0.01, wd=0
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=64, epochs=200, dropout=0.5, lr=0.01, wd=0.001, HyperND_ord=2, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.5, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, lam0=10, lam1=10, alpha=0.1, prop_step=8, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, epochs=100, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=300, dropout=0.5, lr=0.1, wd=0, activation=relu, neg_slope=0.01, gamma=0.01, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, epochs=300, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=2, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, MLP_num_layers=2, epochs=300, dropout=0.7, alpha=0.1, lr=0.001, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=128, epochs=400, dropout=0.5, lr=0.1, wd=0, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=1, MLP_hidden=512, epochs=200, lr=0.01, wd=0, dropout=0.5, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=0.3, lamda=0.1, reduce=mean
AllSetformer     [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=512, MLP_num_layers=2, epochs=100, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=64, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=1, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.2
HyperGT          [dataset section] All_num_layers=1, embedding_hidden=128, hidden_channels=512, epochs=200, dropout=0.5, lr=0.001, wd=0.01, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=500, dropout=0.2, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=8, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=50, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=128, M=5, Mlst=[5], combine=concat
```

### 4.5 Actor (`actor`)

```bash
python main.py --dname=actor --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=actor --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.8, lr=0.01, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.5, lr=0.05, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.02, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=400, dropout=0.5, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=400, dropout=0.5, lr=0.1, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=400, dropout=0.5, lr=0.01, wd=0
LEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.1, lr=0.01, wd=0.0001
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=256, epochs=300, dropout=0.5, lr=0.01, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.8, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0, prop_step=4, normalization=ln, encoder_num_layers=2, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.5, lr=0.01, wd=0, activation=relu, neg_slope=0.01, gamma=0.1, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=1, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=512, MLP_num_layers=3, epochs=100, dropout=0.1, alpha=1, lr=0.01, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=128, epochs=300, dropout=0.5, lr=0.01, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=1, MLP_hidden=256, epochs=200, lr=0.01, wd=0, dropout=0.5, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=1, lamda=0.5, reduce=mean
AllSetformer     [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, MLP_num_layers=2, epochs=200, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=180, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=2, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.2
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.005, wd=0.001, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.2, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=256, M=5, Mlst=[5], combine=concat
```

### 4.6 Twitch (Gamers) (`twitch`)

```bash
python main.py --dname=twitch --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=twitch --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.8, lr=0.01, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.5, lr=0.05, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.02, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0.5, lr=0.001, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0, lr=0.01, wd=0
LEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.1, lr=0.001, wd=0.01
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=256, epochs=300, dropout=0.5, lr=0.01, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.4, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0, prop_step=8, normalization=ln, encoder_num_layers=2, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.1, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0.1, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=80, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=1, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=1024, MLP_num_layers=3, epochs=150, dropout=0.5, alpha=0.1, lr=0.01, wd=0.0001, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.001, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=1, MLP_hidden=128, epochs=100, lr=0.001, wd=0, dropout=0.1, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=0, lamda=0.5, reduce=mean
AllSetformer     [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, MLP_num_layers=2, epochs=200, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=160, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=1, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=2, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.2
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.01, wd=0.01, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.1, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=50, dropout=0.5, lr=0.001, wd=0, hyperG_norm=false, hid_dim=256, M=5, Mlst=[5], combine=concat
```

### 4.7 Pubmed (`pubmed`)

```bash
python main.py --dname=pubmed --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=pubmed --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.8, lr=0.001, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.02, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.5, lr=0.01, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0
LEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0.0001
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=512, epochs=300, dropout=0.2, lr=0.01, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.2, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=64, epochs=300, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0.2, prop_step=16, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=256, epochs=80, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0, activation=relu, neg_slope=0.01, gamma=0.1, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=1, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=512, MLP_num_layers=3, epochs=200, dropout=0.7, alpha=0.05, lr=0.001, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=512, epochs=50, dropout=0.5, lr=0.01, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=false
UniGCNII         [dataset section] All_num_layers=2, MLP_hidden=128, epochs=100, lr=0.001, wd=0, dropout=0, input_drop=0.3, activation=prelu, use_norm=false, restart_alpha=0.5, lamda=0.5, reduce=mean
AllSetformer     [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, MLP_num_layers=2, epochs=300, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=64, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=1, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.8
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.01, wd=0.001, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=true, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=8, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=256, M=5, Mlst=[5], combine=concat
```

### 4.8 Amazon (Ratings) (`amazon`)

```bash
python main.py --dname=amazon --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=amazon --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.8, lr=0.01, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.5, lr=0.05, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.05, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=400, dropout=0.5, lr=0.01, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.5, lr=0.005, wd=0
LEGCN            [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=256, epochs=50, dropout=0.1, lr=0.01, wd=0.01
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=256, epochs=300, dropout=0.5, lr=0.01, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.2, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0, prop_step=4, normalization=ln, encoder_num_layers=2, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=80, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.1, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=1, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, MLP_num_layers=3, epochs=190, dropout=0.5, alpha=0.1, lr=0.01, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=128, epochs=50, dropout=0.5, lr=0.001, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=1, MLP_hidden=128, epochs=50, lr=0.001, wd=0, dropout=0.1, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=0, lamda=0.8, reduce=mean
AllSetformer     [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, MLP_num_layers=2, epochs=200, dropout=0.5, lr=0.01, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=160, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=2, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.1
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=256, epochs=100, dropout=0.5, lr=0.005, wd=0.001, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=140, dropout=0, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=50, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=256, M=5, Mlst=[5], combine=concat
```

### 4.9 Coauthor-DBLP (`coauthor_dblp`)

```bash
python main.py --dname=coauthor_dblp --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=coauthor_dblp --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.8, lr=0.001, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.02, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.8, lr=0.001, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=50, dropout=0.5, lr=0.01, wd=0
LEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.1, lr=0.01, wd=0.01
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=128, epochs=500, dropout=0.5, lr=0.001, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.3, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0.1, prop_step=16, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=4, embedding_hidden=128, MLP_hidden=256, epochs=20, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0.1, sample_ratio=0.1, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=1, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=1, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=1024, MLP_num_layers=3, epochs=50, dropout=0.8, alpha=0.3, lr=0.0006, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=2, MLP_hidden=128, epochs=100, lr=0.001, wd=0, dropout=0.5, input_drop=0.3, activation=prelu, use_norm=false, restart_alpha=0.2, lamda=0.3, reduce=mean
AllSetformer     [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, MLP_num_layers=2, epochs=50, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=40, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=1, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=2, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.7
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.0005, wd=0.01, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=20, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=128, M=5, Mlst=[5], combine=concat
```

### 4.10 Walmart-Trips-100 (`walmart-trips-100`)

```bash
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=walmart-trips-100 --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.8, lr=0.01, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.8, lr=0.02, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0
LEGCN            [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.1, lr=0.01, wd=0.01
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=256, epochs=600, dropout=0.5, lr=0.01, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.3, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0.1, prop_step=16, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [no YAML section] config_sheafhypergnn.yaml has no 'walmart-trips-100' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'All_num_layers'
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0.1, sample_ratio=0.1, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [no YAML section] config_dphgnn.yaml has no 'walmart-trips-100' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'mediator'
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=512, MLP_num_layers=3, epochs=100, dropout=0.7, alpha=0, lr=0.01, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=128, epochs=300, dropout=0.5, lr=0.01, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=2, MLP_hidden=128, epochs=200, lr=0.001, wd=0, dropout=0.5, input_drop=0.3, activation=prelu, use_norm=false, restart_alpha=0, lamda=0.3, reduce=mean
AllSetformer     [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, MLP_num_layers=2, epochs=200, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=200, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=1, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=2, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.7
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.0005, wd=0.01, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=500, dropout=0.2, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [no YAML section] config_tmphn.yaml has no 'walmart-trips-100' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   ArrayMemoryError: unable to allocate 52.6 GiB (dense H)
```

### 4.11 Yelp (`yelp`)

```bash
python main.py --dname=yelp --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=yelp --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.8, lr=0.01, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.01, wd=0, use_bn=false
CEGAT            [no YAML section] config_cegat.yaml has no 'yelp' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'MLP_hidden'
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=1000, dropout=0.5, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=500, dropout=0.2, lr=0.001, wd=0, HyperGCN_mediators=false, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=1, embedding_hidden=128, MLP_hidden=64, epochs=500, dropout=0.5, lr=0.01, wd=0
LEGCN            [no YAML section] config_legcn.yaml has no 'yelp' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   RuntimeError: CPU allocator out of memory (~31.8 GiB in line expansion)
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=128, epochs=500, dropout=0.5, lr=0.001, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.3, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0.1, prop_step=16, normalization=ln, encoder_num_layers=1, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [no YAML section] config_sheafhypergnn.yaml has no 'yelp' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'All_num_layers'
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0.1, sample_ratio=0.1, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [no YAML section] config_dphgnn.yaml has no 'yelp' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'mediator'
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, MLP_num_layers=3, epochs=500, dropout=0.2, alpha=0, lr=0.001, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=3, embedding_hidden=256, MLP_hidden=128, epochs=100, dropout=0.2, lr=0.01, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=2, MLP_hidden=128, epochs=180, lr=0.01, wd=0, dropout=0.2, input_drop=0, activation=prelu, use_norm=false, restart_alpha=0.8, lamda=0.8, reduce=mean
AllSetformer     [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=64, MLP_num_layers=1, epochs=500, dropout=0.2, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=500, dropout=0.2, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=64, MLP_num_layers=1, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=2, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.2
HyperGT          [no YAML section] config_hypergt.yaml has no 'yelp' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'hidden_channels'
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=1000, dropout=0.2, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [no YAML section] config_tmphn.yaml has no 'yelp' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   ArrayMemoryError: unable to allocate 138 GiB (dense H)
```

### 4.12 Pokec (`pokec`)

```bash
python main.py --dname=pokec --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=pokec --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0, lr=0.01, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=100, dropout=0.5, lr=0.05, wd=0, use_bn=false
CEGAT            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=50, dropout=0.8, lr=0.01, wd=0, heads=4, concat=false, use_bn=false
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0, lr=0.01, wd=0
HyperGCN         [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0.5, lr=0.001, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=200, dropout=0, lr=0.005, wd=0
LEGCN            [dataset section] All_num_layers=4, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.1, lr=0.001, wd=0.0001
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=256, epochs=300, dropout=0.5, lr=0.01, wd=1e-05, HyperND_ord=1, HyperND_tol=0.001, HyperND_steps=100, restart_alpha=0.6, normalization=ln, aggregate=sum
PhenomNN         [dataset section] embedding_hidden=128, MLP_hidden=64, epochs=200, dropout=0.5, lr=0.01, wd=1e-05, lam0=10, lam1=10, alpha=0, prop_step=8, normalization=ln, encoder_num_layers=2, decoder_hidden=128, decoder_num_layers=1
SheafHyperGNN    [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=200, dropout=0.5, lr=0.001, wd=0, init_hedge=avg, sheaf_normtype=sym_degree_norm, sheaf_act=sigmoid, sheaf_dropout=false, sheaf_left_proj=false, dynamic_sheaf=true, sheaf_special_head=false, sheaf_pred_block=MLP_var1, sheaf_transformer_head=1, stalk_dim=4, AllSet_input_norm=true, residual_sheaf=false
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.1, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0, sample_ratio=false, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=100, dropout=0.5, lr=0.01, wd=0, expan_dim=256, cg_num_layer=2, cg_MLP_hidden=64, cg_dropout=0.5, hg_num_layer=2, hg_MLP_hidden=64, hg_dropout=0.5, mediator=true, sg_num_layer=2, sg_MLP_hidden=64, sg_dropout=0.5, taa_spatial_dim=64, taa_spectral_dim=64, num_heads=1, chunk_size=-1, spectral_embed_dim=64, fc_dim=64, dff_MLP_hidden=64, dff_num_layers=2, atten_neg_slope=0.2
TFHNN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=1024, MLP_num_layers=3, epochs=150, dropout=0.5, alpha=0.9, lr=0.01, wd=0.0001, normalization=ln
HNHN             [dataset section] All_num_layers=1, embedding_hidden=256, MLP_hidden=128, epochs=200, dropout=0.5, lr=0.01, wd=0.0001, HNHN_alpha=-1.5, HNHN_beta=-0.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=1, MLP_hidden=128, epochs=100, lr=0.001, wd=0, dropout=0, input_drop=0.5, activation=prelu, use_norm=false, restart_alpha=0.5, lamda=0.5, reduce=mean
AllSetformer     [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=64, MLP_num_layers=2, epochs=200, dropout=0.5, lr=0.001, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=128, decoder_num_layers=2
EDHNN            [dataset section] All_num_layers=1, embedding_hidden=128, epochs=140, dropout=0.5, lr=0.001, wd=0, edconv_type=EquivSet, MLP_hidden=512, MLP_num_layers=0, MLP2_num_layers=1, MLP3_num_layers=1, decoder_hidden=256, decoder_num_layer=1, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0.2
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.01, wd=0.01, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=128, epochs=180, dropout=0, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=0, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=50, dropout=0.5, lr=0.01, wd=0, hyperG_norm=false, hid_dim=256, M=5, Mlst=[5], combine=concat
```

### 4.13 Trivago (`trivago`)

```bash
python main.py --dname=trivago --task_type=node_cls --method=MLP --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=CEGCN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=CEGAT --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=HGNN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=HyperGCN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=HCHA --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=LEGCN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=HyperND --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=PhenomNN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=SheafHyperGNN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=HJRL --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=DPHGNN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=TFHNN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=HNHN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=UniGCNII --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=AllSetformer --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=EDHNN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=HyperGT --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=EHNN --num_seeds=5
python main.py --dname=trivago --task_type=node_cls --method=TMPHN --num_seeds=5
```

Effective hyperparameters:

```text
MLP              [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.8, lr=0.05, wd=0, normalization=ln, InputNorm=false
CEGCN            [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=1000, dropout=0.5, lr=0.01, wd=0, use_bn=false
CEGAT            [no YAML section] config_cegat.yaml has no 'trivago' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'MLP_hidden'
HGNN             [dataset section] HCHA_symdegnorm=false, All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.5, lr=0.04, wd=0
HyperGCN         [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=800, dropout=0.5, lr=0.03, wd=0, HyperGCN_mediators=true, HyperGCN_fast=true
HCHA             [dataset section] HCHA_symdegnorm=true, All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=500, dropout=0.5, lr=0.04, wd=0
LEGCN            [dataset section] All_num_layers=3, embedding_hidden=128, MLP_hidden=128, epochs=300, dropout=0.1, lr=0.02, wd=0
HyperND          [dataset section] embedding_hidden=128, MLP_num_layers=2, MLP_hidden=64, epochs=1000, dropout=0.2, lr=0.004, wd=0, HyperND_ord=1, HyperND_tol=0.0001, HyperND_steps=100, restart_alpha=0, normalization=ln, aggregate=sum
PhenomNN         [no YAML section] config_phenomnn.yaml has no 'trivago' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'MLP_hidden'
SheafHyperGNN    [no YAML section] config_sheafhypergnn.yaml has no 'trivago' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'All_num_layers'
HJRL             [dataset section] All_num_layers=2, embedding_hidden=128, MLP_hidden=128, epochs=100, dropout=0.5, lr=0.001, wd=0, activation=relu, neg_slope=0.01, gamma=0.1, sample_ratio=0.1, weight_init=1, pos_weight_thresh=1.0e4
DPHGNN           [no YAML section] config_dphgnn.yaml has no 'trivago' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   AttributeError: 'Namespace' object has no attribute 'mediator'
TFHNN            [dataset section] All_num_layers=128, embedding_hidden=128, MLP_hidden=64, MLP_num_layers=3, epochs=1000, dropout=0.2, alpha=0, lr=0.004, wd=0, normalization=ln
HNHN             [dataset section] All_num_layers=2, embedding_hidden=256, MLP_hidden=128, epochs=1000, dropout=0.5, lr=0.03, wd=0, HNHN_alpha=1, HNHN_beta=1.5, HNHN_nonlinear_inbetween=true
UniGCNII         [dataset section] All_num_layers=2, MLP_hidden=64, epochs=1000, lr=0.01, wd=0, dropout=0.5, input_drop=0.2, activation=relu, use_norm=false, restart_alpha=0, lamda=0.3, reduce=sum
AllSetformer     [dataset section] All_num_layers=4, embedding_hidden=128, MLP_hidden=64, MLP_num_layers=1, epochs=1000, dropout=0.5, lr=0.01, wd=0, aggregate=mean, normalization=ln, deepset_input_norm=true, GPR=true, LearnMask=true, PMA=true, heads=1, output_heads=1, decoder_hidden=64, decoder_num_layers=1
EDHNN            [dataset section] All_num_layers=8, embedding_hidden=128, epochs=1000, dropout=0.5, lr=0.01, wd=0, edconv_type=EquivSet, MLP_hidden=64, MLP_num_layers=0, MLP2_num_layers=0, MLP3_num_layers=1, decoder_hidden=64, decoder_num_layer=1, aggregate=mean, normalization=ln, activation=prelu, AllSet_input_norm=true, alpha=0
HyperGT          [dataset section] All_num_layers=2, embedding_hidden=128, hidden_channels=128, epochs=100, dropout=0.5, lr=0.0005, wd=0.01, num_heads=4, nb_random_features=30, use_bn=true, use_gumbel=true, use_residual=true, use_act=true, use_jk=true, nb_gumbel_sample=10, rb_order=0, rb_trans=sigmoid, use_edge_loss=false, pe=HEPEHtEPE, hefeat=mean
EHNN             [dataset section] All_num_layers=1, embedding_hidden=128, MLP_hidden=256, epochs=500, dropout=0.2, normalization=ln, clip_grad=false, clip_thresh=5, lr=0.001, wd=1e-05, chunk_size=1000, ehnn_ff_layers=2, ehnn_qk_channel=64, ehnn_n_heads=4, ehnn_inner_channel=64, ehnn_hidden_channel=64, ehnn_type=linear, ehnn_pe_dim=64, ehnn_hyper_dim=64, ehnn_hyper_layers=2, ehnn_hyper_dropout=0.2, ehnn_force_broadcast=False, ehnn_input_dropout=0, ehnn_mlp_classifier=True, decoder_hidden=64, decoder_num_layers=2, ehnn_att0_dropout=0, ehnn_att1_dropout=0
TMPHN            [no YAML section] config_tmphn.yaml has no 'trivago' section.
                 method_config() indexes the YAML with the dataset name and raises a
                 KeyError that is caught by a bare except, so no YAML values are loaded.
                 Argparse parser defaults remain (epochs=5, lr=0.001, dropout=0.5,
                 embedding_hidden=128) and the run terminates with:
                   ArrayMemoryError: unable to allocate 261 GiB (dense H)
```

---

## 5. Reproduction Notes

The 28 failed combinations fall into four mechanisms, using the same labels as the study notes:

- **(i) OOM in paper:** the combination is already reported as out-of-memory in the benchmark and cannot fit here either.
- **(ii) Local GPU OOM:** the benchmark reports a score, but the run exhausts the 22 GB GPU memory.
- **(iii) Missing YAML section:** the per-dataset YAML section is absent, so the run aborts with an `AttributeError`. The bare `except` in `method_config()` silently swallows the underlying KeyError and leaves the argparse parser defaults in place.
- **(iv) Host-memory OOM:** a preprocessing step requests a very large CPU array and exceeds the available CPU RAM (for example TMPHN and Yelp/LEGCN). These are host-memory failures, not GPU OOM.

- **UniGCNII:** the benchmark reports only the UniGCNII variant of the UniGNN family, identified as the most competitive variant in the original UniGNN paper.
- Overall, 232 of the 260 combinations completed successfully; the remaining 28 are recorded in both the Excel workbook and the LaTeX tables.

## 6. Source and Modifications

This repository is built on the official DHG-Bench codebase:

- Upstream repository: https://github.com/Coco-Hut/DHG-Bench
- The original project structure, models and data pipeline are retained, and the benchmark paper is cited in Section 7.

Besides adding the reproduction results, the `study_notes/` folder and this README, the following source changes were made. All of them change the internal representation or configuration loading, never the model definition or the intended computation.

**`dhgbench/lib_models/HNN/legcn.py`**

- Every `GCNConv` is created with `add_self_loops=False` (4 call sites), so the line-expansion graph does not get a second self-loop added on top of the one already present in the data.

**`dhgbench/lib_models/HNN/preprocessing.py`**

- `line_expansion()` now wraps the computed adjacency in a transposed `SparseTensor` (`SparseTensor.from_edge_index(...).t()`) instead of returning the raw COO edge index. `GCNConv` then uses the sparse message-passing kernel and never materialises a dense adjacency. On the large Walmart line-expansion graph (about 75.6 million edges) the original path built a 36.07 GiB dense tensor and triggered a CUDA OOM, whereas the sparse representation peaks at 9.35 GiB.
- `ehnn_preprocessing()` adds `coauthor_dblp` to the list of datasets that skip the dense incidence expansion. Like `yelp` and `walmart-trips-100`, `coauthor_dblp` now stays on the sparse path and no longer builds the 41,302 x 63,665 dense matrix (which needed 9.80 GiB of contiguous host memory). The computed result is unchanged, and EHNN then runs at 90.37%.

**`dhgbench/lib_yamls/node_yamls/config_hjrl.yaml`**

- Added the missing per-dataset sections for `coauthor_dblp`, `walmart-trips-100`, `yelp` and `trivago`.

**`dhgbench/lib_yamls/node_yamls/config_unigin.yaml`**

- Added the missing per-dataset sections for all 12 node-classification datasets other than `cora` (the original file only contained `default` and `cora`).

These YAML sections let the config loader return a valid configuration instead of raising an `AttributeError` from a missing key.

## 7. Citation
| **ID** | **Method** | **Paper** |     **Venue**     |  
| :----: | :----: | :----: |:----: |  
|1 | HGNN | [Hypergraph neural networks](https://cdn.aaai.org/ojs/4235/4235-13-7289-1-10-20190705.pdf) | AAAI 2019 | 
|2|HyperGCN|[HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://proceedings.neurips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf)|NeurIPS 2019| 
|3| HNHN | [HNHN: Hypergraph Networks with Hyperedge Neurons](https://grlplus.github.io/papers/40.pdf) | ICML WS 2020 | 
|4| HCHA | [Hypergraph convolution and hypergraph attention](https://www.sciencedirect.com/science/article/abs/pii/S0031320320304404?via=ihub) | PR 2020 | 
|5| UniGNN | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://www.ijcai.org/proceedings/2021/0353.pdf) | IJCAI 2021 | 
|6| AllSetformer | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/forum?id=hpBTIv2uy_E) | ICLR 2022 | 
|7|HyperND|[Nonlinear Feature Diffusion on Hypergraphs](https://proceedings.mlr.press/v162/prokopchik22a/prokopchik22a.pdf)|ICML 2022|
|8| EHNN|[Equivariant hypergraph neural networks](https://arxiv.org/pdf/2208.10428)|ECCV 2022|Y|Y|
|9| LEGCN|[Semi-supervised Hypergraph Node Classification on Hypergraph Line Expansion](https://arxiv.org/pdf/2005.04843)|CIKM 2022|
|10| ED-HNN | [Equivariant Hypergraph Diffusion Neural Operators](https://openreview.net/forum?id=RiTjKoscnNd) | ICLR 2023 | 
|11|PhenomNN|[From Hypergraph Energy Functions to Hypergraph Neural Networks](https://proceedings.mlr.press/v202/wang23d/wang23d.pdf)|ICML 2023|
|12|SheafHyperGNN|[Sheaf Hypergraph Networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/27f243af2887d7f248f518d9b967a882-Paper-Conference.pdf)|NeurIPS 2023|
|13|HJRL|[Hypergraph Joint Representation Learning for Hypervertices and Hyperedges via Cross Expansion](https://openreview.net/forum?id=fxLaL5s6UH)|AAAI 2024|
|14| DPHGNN |[DPHGNN: A Dual Perspective Hypergraph Neural Networks](https://arxiv.org/pdf/2405.16616)|KDD 2024| 
|15|T-HyperGNNs |[T-HyperGNNs: Hypergraph Neural Networks via Tensor Representations](https://ieeexplore.ieee.org/document/10462516)| TNNLS 2024|
|16|HyperGT|[Hypergraph Transformer for Semi-Supervised Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10446248)|ICASSP 2024|
|17|TF-HNN |[Training-Free Message Passing for Learning on Hypergraphs](https://openreview.net/pdf?id=4AuyYxt7A2)| ICLR 2025|

```
