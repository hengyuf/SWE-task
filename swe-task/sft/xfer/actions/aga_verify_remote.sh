# Verify a Data Mover copy on AGA, sent as `aga-ssh "bash -s -- <DEST> <expected file count>" < this file`. Read-only.
# Passes only if every SHA256SUMS entry matches, no file outside SHA256SUMS exists (stale staging objects would land
# here too, and `sha256sum -c` alone does not see them), and the file count equals the dfw source's.
set -u
export LC_ALL=C
DEST="$1" want="${2:-}"
cd "$DEST" 2>/dev/null || { echo "AGA_VERIFY_FAIL no such dir $DEST"; exit 1; }
[[ -s SHA256SUMS ]] || { echo "AGA_VERIFY_FAIL no SHA256SUMS"; exit 1; }
sha256sum --quiet -c SHA256SUMS </dev/null || { echo "AGA_VERIFY_FAIL sha256 mismatch"; exit 1; }
extra=$(comm -23 <(find . -type f ! -name SHA256SUMS | sort) <(sed -E 's/^[0-9a-f]{64}  //' SHA256SUMS | sort))
[[ -z "$extra" ]] || { echo "AGA_VERIFY_FAIL files not in SHA256SUMS: $(echo "$extra" | head -5 | tr '\n' ' ')"; exit 1; }
n=$(find . -type f | wc -l)
[[ -z "$want" || "$n" == "$want" ]] || { echo "AGA_VERIFY_FAIL file count $n, dfw has $want"; exit 1; }
echo "AGA_SHA256_OK files=$n bytes=$(du -sb . | cut -f1) realpath=$(readlink -f .)"
ls -la .
for d in */; do echo "# $d"; ls -la "$d" | head -20; done
