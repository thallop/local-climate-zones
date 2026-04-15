- `finetune.py` → Training script for fine-tuning a pretrained [model](https://github.com/antofuller/CROMA) on So2Sat LCZ42 (urban classes only)
- `inference.ipynb` → Notebook for evaluating the fine-tuned model

```
#croma_pytorch.def
Bootstrap: docker
From: nvidia/cuda:12.1.0-runtime-ubuntu22.04

%post
    apt-get update && apt-get install -y \
        python3 python3-pip python3-dev git \
        libhdf5-dev \
        && rm -rf /var/lib/apt/lists/*

    python3 -m pip install --upgrade pip

    # 1) PyTorch + CUDA depuis l'index PyTorch
    python3 -m pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

    # 2) Autres libs depuis PyPI normal
    python3 -m pip install \
        h5py numpy einops tqdm matplotlib

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export PYTHONUNBUFFERED=1
    export PYTHONIOENCODING=utf-8

%runscript
    exec python3 "$@"


#run_croma_so2sat.slurm
#!/bin/bash
#SBATCH --job-name=croma_so2sat
#SBATCH --partition=gpu-ondemand
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=croma_so2sat_%j.out
#SBATCH --error=croma_so2sat_%j.err

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export PROJ_ROOT=$HOME/work_ird/croma_project
export CODEDIR=$PROJ_ROOT/code
export IMGDIR=$PROJ_ROOT/images
export DATADIR=$HOME/scratch_hallopeaut/LCZ42
export OUTDIR=$HOME/scratch_hallopeaut/croma_runs

mkdir -p "$OUTDIR"

cd "$CODEDIR"

# Exécution dans le conteneur Apptainer avec accès GPU (--nv)
apptainer exec --nv \
  --bind "$PROJ_ROOT","$DATADIR","$OUTDIR" \
  "$IMGDIR/croma-pytorch.sif" \
  python3 finetune.py \
    --train_h5 "$DATADIR/training.h5" \
    --val_h5 "$DATADIR/validation.h5" \
    --test_h5 "$DATADIR/testing.h5" \
    --croma_weights "$CODEDIR/CROMA_base.pt" \
    --batch_size 128 \
    --img_res 120 \
    --num_workers 8 \
    --epochs_phase1 1 \
    --epochs_phase2 10 \
    --device cuda \
    --output_dir "$OUTDIR"

echo "Job finished at $(date)"


Job started on io-gpu-01.io.internal at Wed Dec 17 05:52:59 PM CET 2025
CUDA_VISIBLE_DEVICES=0
Class weights: [2.049051   0.42505795 0.32766196 1.200392   0.6296362  0.2942644
 3.1766872  0.26406425 0.76447225 0.8687126 ]
Initializing SAR encoder
Initializing optical encoder
Initializing joint SAR-optical encoder
[Phase1] 1/1 loss=1.4330 val_loss=1.3956 val_acc=0.5297 val_f1_macro=0.4575
[Phase2] 1/10 loss=0.7421 val_loss=1.0713 val_acc=0.6477 val_f1_macro=0.5767
[Phase2] 2/10 loss=0.4598 val_loss=1.2096 val_acc=0.6809 val_f1_macro=0.6017
[Phase2] 3/10 loss=0.3124 val_loss=1.3480 val_acc=0.6661 val_f1_macro=0.5857
[Phase2] 4/10 loss=0.2071 val_loss=1.5144 val_acc=0.6482 val_f1_macro=0.5869
[Phase2] 5/10 loss=0.1372 val_loss=1.6541 val_acc=0.6554 val_f1_macro=0.5944
[Phase2] 6/10 loss=0.0889 val_loss=2.1943 val_acc=0.6456 val_f1_macro=0.5838
[Phase2] 7/10 loss=0.0685 val_loss=1.9845 val_acc=0.6650 val_f1_macro=0.6006
[Phase2] 8/10 loss=0.0504 val_loss=2.2742 val_acc=0.6333 val_f1_macro=0.5768
[Phase2] 9/10 loss=0.0367 val_loss=2.4132 val_acc=0.6115 val_f1_macro=0.5526
[Phase2] 10/10 loss=0.0290 val_loss=2.5892 val_acc=0.6593 val_f1_macro=0.5972
Best model (F1_macro=0.6017) saved to /home/hallopeaut/scratch_hallopeaut/croma_runs/croma_lcz42_best_f1_macro.pth
[TEST] loss=2.5624 acc=0.6844 f1_macro=0.6223
Job finished at Wed Dec 17 09:47:21 PM CET 2025
```