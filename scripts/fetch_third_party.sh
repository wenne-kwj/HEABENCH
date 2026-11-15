#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
third_party_root="${repository_root}/third_party"
mkdir -p "${third_party_root}"

fetch_pinned() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="$4"

  if [[ -e "${destination}" ]]; then
    if [[ ! -d "${destination}/.git" ]]; then
      echo "ERROR: ${destination} exists but is not a Git checkout" >&2
      return 1
    fi
    local actual
    actual="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual}" != "${commit}" ]]; then
      echo "ERROR: ${name} is at ${actual}, expected ${commit}; refusing to reset it" >&2
      return 1
    fi
    echo "${name}: already pinned at ${commit}"
    return 0
  fi

  echo "${name}: fetching ${commit}"
  git init -q "${destination}"
  git -C "${destination}" remote add origin "${url}"
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" -c advice.detachedHead=false checkout --detach FETCH_HEAD
  local actual
  actual="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "ERROR: ${name} checkout verification failed" >&2
    return 1
  fi
}

fetch_pinned \
  "CGCNN" \
  "https://github.com/txie-93/cgcnn.git" \
  "f42ab233c4ee0c416879d6bc2d22a264418413ad" \
  "${third_party_root}/CGCNN"

fetch_pinned \
  "ALIGNN" \
  "https://github.com/usnistgov/alignn.git" \
  "f2366daa3413d28a825b46e34d001b5549b05a40" \
  "${third_party_root}/ALIGNN"

fetch_pinned \
  "ComFormer/AIRS" \
  "https://github.com/divelab/AIRS.git" \
  "4a16c68a7da707c521019067dec51c227c10de45" \
  "${third_party_root}/ComFormer"

fetch_pinned \
  "EDBench" \
  "https://github.com/HongxinXiang/EDBench.git" \
  "89183fee107501589d2c2ab3c8be8ab54096e0f6" \
  "${third_party_root}/EDBench"

fetch_pinned \
  "GeoFormer" \
  "https://github.com/microsoft/AI2BMD.git" \
  "20ba4c1fe3045de862f03ddbb84c10811884f5f8" \
  "${third_party_root}/geoformer"

fetch_pinned \
  "EquiformerV2" \
  "https://github.com/atomicarchitects/equiformer_v2.git" \
  "d5ad4be729b56f74012ebb7f097f77c5b00a1004" \
  "${third_party_root}/equiformer_v2"

fetch_pinned \
  "PointVector/OpenPoints" \
  "https://github.com/guochengqian/openpoints.git" \
  "69fd158bc4951c0c99e976d1e210cba8935208de" \
  "${third_party_root}/openpoints_pointvector"

fetch_pinned \
  "X-3D" \
  "https://github.com/sunshuofeng/X-3D.git" \
  "d1648995179391be17502bb6f5a9c5320ef318a3" \
  "${third_party_root}/x3d"

fetch_pinned \
  "Pointcept/PTv3" \
  "https://github.com/Pointcept/Pointcept.git" \
  "2b247ed555e96eae11710dfde79a3fc16c8c74b3" \
  "${third_party_root}/pointcept_ptv3"

echo "All third-party source commits verified."
