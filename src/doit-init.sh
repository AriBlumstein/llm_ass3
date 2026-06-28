# doit shell integration — enables persistent `cd` (and other shell-state) navigation.
#
# A normal program runs as a child process and CANNOT change your shell's working directory.
# Sourcing this file defines a `doit` shell function that runs the agent, then performs any
# directory change the agent requested *in your current shell* — so `doit "go to my project"`
# actually moves your terminal.
#
# Install: add this line to your ~/.bashrc or ~/.zshrc (adjust the path):
#     source /ABSOLUTE/PATH/TO/llm_ass3/src/doit-init.sh
# Then open a new shell (or `source` your rc file) and use `doit` as usual.
#
# Works in bash and zsh.

# Resolve the directory of THIS script (bash and zsh differ in how they expose it).
if [ -n "${BASH_SOURCE:-}" ]; then
    _DOIT_SELF="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _DOIT_SELF="${(%):-%x}"
else
    _DOIT_SELF="$0"
fi
_DOIT_LAUNCHER="$(cd "$(dirname "$_DOIT_SELF")" && pwd)/doit"
unset _DOIT_SELF

doit() {
    # Two out-of-band channels back to THIS shell:
    #   _doit_cdfile : a target directory (for `cd`, hoisted as a quoted value)
    #   _doit_shfile : a session-state builtin command (export/alias/set/unset/shopt/pushd/popd)
    local _doit_cdfile _doit_shfile _doit_rc
    _doit_cdfile="$(mktemp "${TMPDIR:-/tmp}/doit-cd.XXXXXX")" || {
        command "$_DOIT_LAUNCHER" "$@"      # mktemp failed -> run without shell integration
        return $?
    }
    _doit_shfile="$(mktemp "${TMPDIR:-/tmp}/doit-sh.XXXXXX")" || _doit_shfile=""

    # Run the agent. Its normal output and interactive prompts (clarification, [y/N]) go straight
    # to the terminal — only the shell-state directives come back via the temp files, so we do NOT
    # capture stdout and interactivity is fully preserved.
    DOIT_CD_FILE="$_doit_cdfile" DOIT_SHELL_FILE="$_doit_shfile" \
        command "$_DOIT_LAUNCHER" "$@"
    _doit_rc=$?

    # Apply a directory change in THIS shell so it persists.
    if [ -s "$_doit_cdfile" ]; then
        cd "$(cat "$_doit_cdfile")"
    fi
    # Apply a session-state builtin in THIS shell. The command was screened by the agent (no
    # command substitution, chaining, piping, or redirection), so it can only mutate shell state.
    if [ -n "$_doit_shfile" ] && [ -s "$_doit_shfile" ]; then
        eval "$(cat "$_doit_shfile")"
    fi

    rm -f "$_doit_cdfile" "$_doit_shfile"
    return $_doit_rc
}
