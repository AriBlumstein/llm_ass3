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

# --- User-command recorder (accurate success/failure for user awareness) ---------------------------
# A per-command hook logs "<exit status>\t<command>" to a per-session file, so doit knows not just
# WHAT you ran in the terminal but whether it SUCCEEDED or FAILED (the shell history alone has no exit
# status). doit reads this file via $DOIT_CMD_LOG; it falls back to `fc -l` (no exit status) if absent.
if [ -z "${DOIT_CMD_LOG:-}" ]; then
    _DOIT_REPO="$(cd "$(dirname "$_DOIT_LAUNCHER")/.." 2>/dev/null && pwd)"
    if [ -n "$_DOIT_REPO" ]; then
        mkdir -p "$_DOIT_REPO/.doit" 2>/dev/null
        export DOIT_CMD_LOG="$_DOIT_REPO/.doit/cmdlog_$$.tsv"
    fi
    unset _DOIT_REPO
fi

# Pairing the command TEXT with its exit STATUS needs the preexec pattern: capture the command just
# BEFORE it runs (so we have its text), then log it with $? just AFTER (the prompt hook). Doing it the
# other way - reading history in the prompt hook - mis-pairs, because bash adds a command to history
# only AFTER PROMPT_COMMAND, so the history lags the status by one.
__doit_pending=""
__doit_preexec() {                                  # DEBUG trap: $BASH_COMMAND = command about to run
    case "$BASH_COMMAND" in *__doit_*) return ;; esac   # ignore our own machinery
    __doit_pending="$BASH_COMMAND"
}
__doit_record() {                                   # prompt hook: pair the captured command with $?
    local _ec=$?
    if [ -n "${DOIT_CMD_LOG:-}" ] && [ -n "${__doit_pending:-}" ]; then
        printf '%s\t%s\n' "$_ec" "$__doit_pending" >> "$DOIT_CMD_LOG" 2>/dev/null
    fi
    __doit_pending=""
}
if [ -n "${ZSH_VERSION:-}" ]; then
    # zsh has native preexec (gets the command line as $1) and precmd hooks - cleaner than bash.
    __doit_zpreexec() { __doit_pending="$1"; }
    autoload -Uz add-zsh-hook 2>/dev/null && {
        add-zsh-hook preexec __doit_zpreexec
        add-zsh-hook precmd  __doit_record
    }
else
    case ";${PROMPT_COMMAND:-};" in
        *";__doit_record;"*) ;;                                          # already registered
        *) PROMPT_COMMAND="__doit_record${PROMPT_COMMAND:+;$PROMPT_COMMAND}" ;;
    esac
    # Set the DEBUG trap LAST, then clear pending, so this file's own setup commands aren't recorded.
    __doit_pending=""
    trap '__doit_preexec' DEBUG
fi
# ---------------------------------------------------------------------------------------------------

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

    # Capture the user's recent shell commands from the LIVE in-memory history (`fc -l`, normalized
    # across bash/zsh). This is what makes doit aware of what YOU did in the terminal (user
    # awareness), separate from doit's own actions.
    local _doit_hist
    _doit_hist="$(fc -l -50 2>/dev/null)"

    # Run the agent. Its normal output and interactive prompts (clarification, [y/N]) go straight
    # to the terminal — only the shell-state directives come back via the temp files, so we do NOT
    # capture stdout and interactivity is fully preserved.
    DOIT_CD_FILE="$_doit_cdfile" DOIT_SHELL_FILE="$_doit_shfile" DOIT_SHELL_HISTORY="$_doit_hist" \
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
