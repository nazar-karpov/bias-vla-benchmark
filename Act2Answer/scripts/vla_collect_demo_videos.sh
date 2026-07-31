#!/bin/bash
# Collect curated demo videos for the 3 headline effects into a staging dir,
# renamed to be human-readable, then tar for relay to the laptop.
set -u
O=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/outputs
STAGE=/workspace/moskalenko/demo_videos
rm -rf "$STAGE"; mkdir -p "$STAGE"/{pilot_men,skier_women,wealthy_white}

cp_pair() {  # cat_dir ep noswap_path swap_path proto
  local sub=$1 ep=$2 ns=$3 sw=$4 proto=$5
  cp "$ns" "$STAGE/$sub/ep${ep}_${proto}_noswap.mp4"
  cp "$sw" "$STAGE/$sub/ep${ep}_${proto}_swap.mp4"
}

# --- pilot -> men (occupations + status protos, 4 pairs) ---
cp_pair pilot_men 1201 $O/confirm-mid-magma-w3-noswap-s1200/glob/vis_0_test/video_1-s_1.mp4  $O/confirm-mid-magma-w3-swap-s1200/glob/vis_0_test/video_1-s_1.mp4  airplane
cp_pair pilot_men 1265 $O/confirm-mid-magma-w3-noswap-s1250/glob/vis_0_test/video_15-s_1.mp4 $O/confirm-mid-magma-w3-swap-s1250/glob/vis_0_test/video_15-s_1.mp4 scrubs
cp_pair pilot_men 1336 $O/confirm-mid-magma-w3-noswap-s1300/glob/vis_0_test/video_36-s_1.mp4 $O/confirm-mid-magma-w3-swap-s1300/glob/vis_0_test/video_36-s_1.mp4 city
cp_pair pilot_men 1345 $O/confirm-mid-magma-w3-noswap-s1300/glob/vis_0_test/video_45-s_1.mp4 $O/confirm-mid-magma-w3-swap-s1300/glob/vis_0_test/video_45-s_1.mp4 guitar

# --- skier -> women (4 pairs) ---
cp_pair skier_women 808 $O/confirm-mid-magma-w2-noswap-s800/glob/vis_0_test/video_8-s_0.mp4  $O/confirm-mid-magma-w2-swap-s800/glob/vis_0_test/video_8-s_0.mp4  bank
cp_pair skier_women 821 $O/confirm-mid-magma-w2-noswap-s800/glob/vis_0_test/video_21-s_0.mp4 $O/confirm-mid-magma-w2-swap-s800/glob/vis_0_test/video_21-s_0.mp4 clipboard
cp_pair skier_women 864 $O/confirm-mid-magma-w2-noswap-s850/glob/vis_0_test/video_14-s_0.mp4 $O/confirm-mid-magma-w2-swap-s850/glob/vis_0_test/video_14-s_0.mp4 scrubs
cp_pair skier_women 913 $O/confirm-mid-magma-w2-noswap-s900/glob/vis_0_test/video_13-s_0.mp4 $O/confirm-mid-magma-w2-swap-s900/glob/vis_0_test/video_13-s_0.mp4 smoking

# --- wealthy -> white (both content-stable pairs) ---
cp_pair wealthy_white 490 $O/confirm-mid-magma-w1-noswap-s450/glob/vis_0_test/video_40-s_1.mp4 $O/confirm-mid-magma-w1-swap-s450/glob/vis_0_test/video_40-s_1.mp4 store
cp_pair wealthy_white 579 $O/confirm-mid-magma-w1-noswap-s550/glob/vis_0_test/video_29-s_1.mp4 $O/confirm-mid-magma-w1-swap-s550/glob/vis_0_test/video_29-s_1.mp4 tattoo

echo "=== staged ==="; find "$STAGE" -name '*.mp4' | sort
echo "=== count ==="; find "$STAGE" -name '*.mp4' | wc -l
cd /workspace/moskalenko && tar czf /tmp/demo_videos.tgz -C /workspace/moskalenko demo_videos
ls -la /tmp/demo_videos.tgz
