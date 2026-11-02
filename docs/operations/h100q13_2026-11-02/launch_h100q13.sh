#!/usr/bin/env bash
# 2026-11-02: Nazar dedicated job 6d3bada7 (4 H100) to the full bias program.
# Existing h100q10 GPU2 reservation remains untouched.
export COND_LIST="focus_q193fa:noswap focus_q193fa:swap focus_q193fb:noswap focus_q193fb:swap"
export NODE_TAG=h100q13
export GPUS="0 1 2 3"
export MODELS="internvla spatialvla xiaomi gr00t"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
exec bash $HOME/ws/_q193f_driver_h100q13.sh
