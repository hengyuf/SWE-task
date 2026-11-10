# probe1: read-only look at the HSG SFT pipeline (report + a tarball of its small text files), run by hsg_step.sbatch.
set -uo pipefail
out="$1"
H=/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/.hsg
A=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/xfer/actions
P=/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/zhenghaox/swe-rubric-pipeline/super-filtered-glm53-v5-20260922
echo "[$(date -Is)] report"
timeout 1800 "$H/hsg-ssh" 'bash -s' < "$A/hsg_probe_remote.sh" > "$out/report.txt" 2> "$out/report.err"
echo "report rc=$? bytes=$(stat -c %s "$out/report.txt")"
echo "[$(date -Is)] text files tarball"
# One tarball (one local inode) of the pipeline's small text files: scripts, configs, manifests, docs, code.
timeout 1800 "$H/hsg-ssh" "cd '$P' && find . -maxdepth 5 \\( -name 'iter_*' -o -name checkpoints -o -name .git -o -name __pycache__ -o -name 'megatron-lm*' -o -name 'Megatron-LM*' \\) -prune -o -type f -size -2M \\( -name '*.sbatch' -o -name '*.sh' -o -name '*.md' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.py' -o -name '*.txt' -o -name '*.tsv' -o -name '*.jinja' -o -name '*.toml' -o -name '*.cfg' -o -name '*.csv' \\) -print0 | head -z -n 4000 | tar --null -czf - -T -" > "$out/pipeline-text.tgz" 2> "$out/tar.err"
echo "tar rc=$? bytes=$(stat -c %s "$out/pipeline-text.tgz")"
tar -tzf "$out/pipeline-text.tgz" > "$out/pipeline-text.list" 2>/dev/null
echo "files in tarball: $(wc -l < "$out/pipeline-text.list")"
