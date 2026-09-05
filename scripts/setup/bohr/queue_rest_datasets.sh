#!/usr/bin/env bash
M=/home/moskalenko/ws/datasets/gdrive_meta
bash $HOME/ws/setup/run_dataset_frames.sh pairs_frames none $HOME/ws/datasets/pairs_dataset $M/pairs_dataset/pairs/gender.tsv,$M/pairs_dataset/pairs/skin_color.tsv > $HOME/ws/logs_pairs_frames.log 2>&1
bash $HOME/ws/setup/run_dataset_frames.sh veri_frames pad $HOME/ws/datasets/veri_emergency $M/veri_emergency/pairs.tsv $M/veri_emergency/deprecated/vla_manifests/veri_two_image_selection.csv $M/veri_emergency/deprecated/vlm_manifests/veri_vlm_parallel_two_image_selection.csv > $HOME/ws/logs_veri_frames.log 2>&1
bash $HOME/ws/setup/run_dataset_frames.sh visbias_frames face $HOME/ws/datasets/visbias $M/visbias/pairs/gender.tsv,$M/visbias/pairs/ethnicity.tsv,$M/visbias/pairs/profession.tsv $M/visbias/deprecated/vla_manifests/visbias_two_image_selection.csv $M/visbias/deprecated/vlm_manifests/visbias_vlm_parallel_two_image_selection.csv > $HOME/ws/logs_visbias_frames.log 2>&1
echo QUEUE_REST_DONE
