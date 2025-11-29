#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -P gcc50441
#PBS -m b
#PBS -m e
#PBS -j oe
#PBS -o log_abci_runs_transe.txt

cd ${PBS_O_WORKDIR}

# 追加: スクリプト名と引数をコマンドライン引数で受け取る（デフォルト: runs_conve.sh）
SCRIPT=${1:-main.sh}
shift
SCRIPT_ARGS="$@"

source /etc/profile.d/modules.sh
source /home/acg16558pn/programs/Simple-Active-Refinement-for-Knowledge-Graph/.venv/bin/activate
module load cuda/12.6/12.6.1
module load cudnn/9.5/9.5.1
echo "[INFO] Job started on: $(date)"
echo "[INFO] Working directory: $(pwd)"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Running script: ./$SCRIPT $SCRIPT_ARGS"
echo "[INFO] PBS_O_WORKDIR: ${PBS_O_WORKDIR}"

# スクリプトを引数付きで実行し、ログファイル名もスクリプト名・日付で動的に
LOGFILE="log_$(basename $SCRIPT)_$(date +%Y%m%d_%H%M%S).txt"
#./$SCRIPT $SCRIPT_ARGS >& $LOGFILE
./$SCRIPT >& $LOGFILE
echo "[INFO] Job finished on: $(date)"