#!/usr/bin/env bash
# Run the 38 real-client checks used for the develop constant-potential patch.
set -euo pipefail
CP_TEST_DIRECTORY="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 || $# -gt 2 || -e "$1" || ( $# == 2 && "$2" != --collect-only ) ]]; then
    echo "Usage: $0 NEW_OUTPUT_DIRECTORY [--collect-only]" >&2
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
CP_PYTEST_OPTIONS=()
if [[ ${2:-} == --collect-only ]]; then
    CP_PYTEST_OPTIONS+=(--collect-only)
fi
python -m pytest -p no:cacheprovider -v "${CP_PYTEST_OPTIONS[@]}" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_charge_and_position_sequence" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_workfunction_matches_cube_on_triclinic_cell" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_ordinary_socket_default_symmetry_and_multiple_positions" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_zero_request_exit" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_invalid_requests_fail_without_results[missing_charge]" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_invalid_requests_fail_without_results[unconverged]" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_ordinary_scf_matches_independent_reference" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_native_solvent_updates_on_every_changed_request" \
    "$CP_TEST_DIRECTORY/test_socket_driver.py::test_kpoint_pools_preserve_charge_and_potential" \
    "$CP_TEST_DIRECTORY/test_combined_and_native.py::test_native_dynamics_matches_ordinary_build" \
    "$CP_TEST_DIRECTORY/test_ase_socket.py" \
    "$CP_TEST_DIRECTORY/test_socket_session_failures.py" \
    "$CP_TEST_DIRECTORY/test_ipi_dynamics.py::test_real_dynamics_and_force_count" \
    "$CP_TEST_DIRECTORY/test_ipi_dynamics.py::test_workfunction_restart" \
    --basetemp="${CP_TEST_OUTPUT}/work" --junitxml="${CP_TEST_OUTPUT}/results.xml" \
    > "${CP_TEST_OUTPUT}/pytest.log" 2>&1
echo "Results: ${CP_TEST_OUTPUT}/pytest.log"
