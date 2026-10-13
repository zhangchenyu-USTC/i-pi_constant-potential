#!/usr/bin/env bash
# Run real driver tests with an independent ordinary reference executable.
set -euo pipefail
CP_TEST_DIRECTORY="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# != 1 || -e "$1" ]]; then
    echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
mkdir -p "$1"
CP_TEST_OUTPUT="$(builtin cd "$1" && pwd)"
export ABACUS_CP_EXECUTABLE="${ABACUS_CP_EXECUTABLE:?Set the modified ABACUS executable}"
export ABACUS_REFERENCE_EXECUTABLE="${ABACUS_REFERENCE_EXECUTABLE:?Set an independent ordinary ABACUS reference executable}"
export ABACUS_PP_ORB="${ABACUS_PP_ORB:-${ABACUS_SOURCE_ROOT:?Set ABACUS_PP_ORB or ABACUS_SOURCE_ROOT}/tests/PP_ORB}"
export ABACUS_TEST_IPI_ROOT="${ABACUS_TEST_IPI_ROOT:-${IPI_ROOT:-$(builtin cd "${CP_TEST_DIRECTORY}/../../../../.." && pwd)}}"
export ABACUS_TEST_RANKS="${ABACUS_TEST_RANKS:-4}"
export PYTHONPATH="${ABACUS_TEST_IPI_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ABACUS_REQUIRE_REAL_TESTS=1
export ABACUS_TEST_IDENTITY_FILE="${CP_TEST_OUTPUT}/identities.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m pytest -p no:cacheprovider -v "${CP_TEST_DIRECTORY}" \
    --basetemp="${CP_TEST_OUTPUT}/work" --junitxml="${CP_TEST_OUTPUT}/results.xml" \
    > "${CP_TEST_OUTPUT}/pytest.log" 2>&1
