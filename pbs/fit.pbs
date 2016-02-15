#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=48:00:00
#PBS -l mem=60GB
#PBS -N pl-fit
#PBS -M danfm@nyu.edu
#PBS -j oe

module purge
export PATH="$HOME/miniconda3/bin:$PATH"
module load mvapich2/intel/2.0rc1

export PEERLESS_DATA_DIR=$SCRATCH/peerless/scratch
export OMP_NUM_THREADS=1

SRCDIR=$HOME/projects/peerless
RUNDIR=$PBS_O_WORKDIR

cd $RUNDIR
mpiexec -np $PBS_NP python $SRCDIR/scripts/peerless-fit $RUNDIR/init.pkl --nwalkers $((PBS_NP*2))

rm -f $RUNDIR/submitted.lock

