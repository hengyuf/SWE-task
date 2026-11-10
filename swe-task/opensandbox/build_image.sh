#!/usr/bin/env bash
# Run inside an existing rootless BuildKit/Gantry runner with registry credentials.
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "Usage: bash build_image.sh CONTEXT IMAGE_REF METADATA_JSON" >&2
    exit 2
fi
context=$(cd -- "$1" && pwd -P)
image_ref=$2
metadata=$3
[[ -f "$context/Dockerfile" ]] || { echo "Context has no Dockerfile" >&2; exit 2; }
[[ ! -e "$metadata" ]] || { echo "Metadata output already exists" >&2; exit 2; }
case "$image_ref" in
    ''|*[!A-Za-z0-9._:/-]*) echo "Image reference contains unsupported characters" >&2; exit 2 ;;
esac
case "${image_ref##*/}" in
    *:*) repository=${image_ref%:*} ;;
    *) echo "Provide an explicit destination image tag" >&2; exit 2 ;;
esac
command -v buildctl-daemonless.sh >/dev/null || { echo "Run this in the existing BuildKit/Gantry runner" >&2; exit 2; }
[[ -n ${DOCKER_CONFIG:-} && -r "$DOCKER_CONFIG/config.json" ]] || {
    echo "The runner must provide DOCKER_CONFIG with registry push credentials" >&2
    exit 2
}
mkdir -p -- "$(dirname -- "$metadata")"
buildctl-daemonless.sh build \
    --frontend dockerfile.v0 \
    --local "context=$context" \
    --local "dockerfile=$context" \
    --opt platform=linux/amd64 \
    --output "type=image,name=$image_ref,push=true" \
    --metadata-file "$metadata"

# BuildKit writes this JSON object itself. No registry credentials are printed.
digest=$(sed -n 's/^[[:space:]]*"containerimage.digest":[[:space:]]*"\(sha256:[a-f0-9]\{64\}\)".*/\1/p' "$metadata")
[[ $digest =~ ^sha256:[a-f0-9]{64}$ ]] || {
    echo "BuildKit completed without a valid published image digest; inspect its metadata" >&2
    exit 1
}
printf '%s@%s\n' "$repository" "$digest"
