#!/usr/bin/env bash
# Перенос /workspace/moskalenko (cloud.ru CPU-нода) -> Selectel Bohr (~/ws). 05.09.2026.
# conda НЕ копируем (абсолютный префикс, на Bohr ставится заново из env_exports/*.yml).
set -u
K=/workspace/moskalenko/.ssh/bohr_ed25519
DST=moskalenko@176.114.85.176
SSH="ssh -i $K -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
R="/workspace/moskalenko/conda/bin/rsync -aH --partial --info=progress2 --no-inc-recursive -e"
LOG=/workspace/moskalenko/logs_migrate_bohr.log
cd /workspace/moskalenko
{
echo "=== START $(date -u)"
echo "--- 1/4 репа (39G: git + outputs + внешние клоны)"
$R "$SSH" bias-vla-benchmark-main $DST:ws/ 2>&1 | tail -3
echo "--- 2/4 мелочь: ассеты ManiSkill, датасеты, PAIRS, экспорт env, setup-скрипты, swiftshader"
$R "$SSH" maniskill_assets datasets PAIRS PAIRS.tar.gz env_exports mesa_vk swiftshader *.sh *.py *.md $DST:ws/ 2>&1 | tail -3
echo "--- 3/4 hf_cache (202G)"
$R "$SSH" hf_cache $DST:ws/ 2>&1 | tail -3
echo "--- 4/4 контроль размеров"
du -sh bias-vla-benchmark-main hf_cache datasets maniskill_assets 2>/dev/null
$SSH $DST 'du -sh ws/bias-vla-benchmark-main ws/hf_cache ws/datasets ws/maniskill_assets 2>/dev/null; df -h / | tail -1'
echo "=== MIGRATE_DONE $(date -u)"
} > $LOG 2>&1
