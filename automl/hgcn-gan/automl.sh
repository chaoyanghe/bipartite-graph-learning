#!/usr/bin/env bash

rm -rf ./../../out/hgcn-gan/tencent/*
cd ./../..
sbatch /automl/hgcn-gan/mpi_run.slurm