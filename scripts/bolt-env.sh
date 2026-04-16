#!/usr/bin/env bash
# bolt-env.sh — source once per bash session on Windows to load MSVC env.
#
# Captures vcvars64 environment and exports INCLUDE/LIB/PATH/LIBPATH into
# the current shell. After this, you can run `ninja -C build/ninja-msvc`
# or `cl.exe` directly — ~1s incremental rebuilds.
#
# Usage:
#   source scripts/bolt-env.sh
#
# Idempotent — running twice is fine.

if [[ -n "$BOLT_MSVC_LOADED" ]]; then
    return 0 2>/dev/null || exit 0
fi

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_dump="$_here/_dump_vcvars_env.cmd"
if [[ ! -f "$_dump" ]]; then
    echo "[bolt-env] missing $_dump" >&2
    return 1 2>/dev/null || exit 1
fi

# Run the dump script in cmd.exe; capture printed env. This avoids
# cross-shell quoting headaches with the vcvars path.
_tmp=$(mktemp)
cmd.exe //c "$(cygpath -w "$_dump")" > "$_tmp" 2>/dev/null

while IFS='=' read -r k v; do
    # Windows env values may contain trailing \r — strip it.
    v="${v%$'\r'}"
    case "$k" in
        INCLUDE|LIB|LIBPATH|Path|PATH)
            # Convert semicolon-separated Windows paths; leave for cl.exe as-is
            # except PATH which bash needs in Unix form.
            if [[ "$k" == "Path" || "$k" == "PATH" ]]; then
                # Convert ; to : and any C:\ to /c/ via cygpath-style.
                unix_path=$(echo "$v" | sed -e 's/\\/\//g' -e 's/;/:/g' \
                    -e 's|\([A-Za-z]\):|/\L\1|g')
                export PATH="$unix_path:$PATH"
            else
                export "$k"="$v"
            fi
            ;;
    esac
done < "$_tmp"
rm -f "$_tmp"

export BOLT_MSVC_LOADED=1
echo "[bolt-env] MSVC env loaded (cl.exe on PATH, INCLUDE/LIB set)"
