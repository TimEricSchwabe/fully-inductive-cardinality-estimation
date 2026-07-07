#!/usr/bin/env bash
# Build G-Care statistics for a single dataset.
# Usage: custom-run-build.sh <dataset>
# GCARE_HOME must be set (or defaults to the gcare/ dir next to this script).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCARE_HOME="${GCARE_HOME:-${SCRIPT_DIR}/gcare}"

data=$1
seed=0
commit_id="12345"
now="$(date +'%Y%m%d%H%M')"
result_dir="${GCARE_HOME}/data/result/build/${now}/"

mkdir -p "${result_dir}"

p=0.03

input_dir="${GCARE_HOME}/data/dataset/${data}/"
data_dir="${GCARE_HOME}/data/dataset/${data}/"

build_methods_csv="${GCARE_BUILD_METHODS:-wj,cset,sumrdf}"
IFS=',' read -r -a build_methods <<< "${build_methods_csv}"

for method in "${build_methods[@]}" ; do

  method="$(echo "${method}" | xargs)"
  if [ -z "${method}" ]; then
    continue
  fi

  if [ "${method}" == "cs" ] || [ "${method}" == "bsk" ]; then
    bin="${GCARE_HOME}/build/gcare_relation"
  else
    bin="${GCARE_HOME}/build/gcare_graph"
  fi

  if [ "${method}" == "bsk" ]; then
    export GCARE_BSK_BUDGET=4096
  fi

  if [ "${method}" == "sumrdf" ]; then
    export GCARE_SUMRDF_THRESHOLD=0.15
  fi

  output_file="${result_dir}/${data}_${method}_p${p}_s${seed}_build_result.txt"

  cmd="${bin} -b -m ${method} -i ${input_dir}/${data}.txt -d ${data_dir}/${data} -p ${p} -s ${seed} -o ${output_file}"
  echo "${cmd}"
  ${cmd}

  echo "#${commit_id}" >> "${output_file}"
done
