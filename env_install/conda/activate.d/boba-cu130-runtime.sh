# Boba's CUDA 13 runtime libraries must precede host copies. Resolve every path
# from the active environment, but not ${CONDA_PREFIX}/lib: its libtinfo
# shadows the host copy.
if [ "${_BOBA_CU130_RUNTIME_ACTIVE:-0}" != "1" ] && [ -n "${CONDA_PREFIX:-}" ]; then
    if [ "${LD_LIBRARY_PATH+x}" = "x" ]; then
        export _BOBA_CU130_LD_LIBRARY_PATH_WAS_SET=1
        export _BOBA_CU130_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
    else
        export _BOBA_CU130_LD_LIBRARY_PATH_WAS_SET=0
        unset _BOBA_CU130_SAVED_LD_LIBRARY_PATH
    fi

    _BOBA_CU130_SITE_PACKAGES="$("${CONDA_PREFIX}/bin/python" -c \
        'import sysconfig; print(sysconfig.get_path("platlib"))')"
    _BOBA_CU130_RUNTIME_PATHS="${_BOBA_CU130_SITE_PACKAGES}/nvidia/cu13/lib:${_BOBA_CU130_SITE_PACKAGES}/torch/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib"

    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="${_BOBA_CU130_RUNTIME_PATHS}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${_BOBA_CU130_RUNTIME_PATHS}"
    fi
    export _BOBA_CU130_RUNTIME_ACTIVE=1

    unset _BOBA_CU130_SITE_PACKAGES
    unset _BOBA_CU130_RUNTIME_PATHS
fi
