# verify-v1: sha256 check of corpus v1 on HSG (retries a refused ssh), run by hsg_step.sbatch.
set -uo pipefail
out="$1"
DEST=/lustre/fsw/portfolios/nemotron/users/hengyuf/sft-kimi-pass-v1-20261023
H=/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/.hsg
for attempt in 1 2 3 4 5 6; do
  timeout 1800 "$H/hsg-ssh" "cd '$DEST' && sha256sum --quiet -c SHA256SUMS && echo HSG_SHA256_OK; ls -la '$DEST' '$DEST/raw_sft' '$DEST/blends'; readlink -f '$DEST'" > "$out/verify.txt" 2>&1
  grep -q HSG_SHA256_OK "$out/verify.txt" && { cat "$out/verify.txt"; echo "[$(date -Is)] verified (attempt $attempt)"; exit 0; }
  echo "[$(date -Is)] attempt $attempt failed: $(tail -2 "$out/verify.txt" | tr '\n' ' ')"
  sleep $((60 * attempt))
done
cat "$out/verify.txt"; exit 1
