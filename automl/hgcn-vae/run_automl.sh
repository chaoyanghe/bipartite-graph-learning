#!/usr/bin/env bash

rm -rf ./../../out/hgcn-vae/tencent/*
cd ./../..
sbatch ./automl/hgcn-vae/mpi_run.slurm