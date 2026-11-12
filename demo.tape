# retrieval-evaluator terminal demo
#
# Before recording:
#   uv sync --extra hf --extra dashboard
#   source .venv/bin/activate
#   Pre-download the HuggingFace model used below.
#
# Record:
#   vhs demo.tape

Output docs/assets/demo-terminal.gif
Output docs/assets/demo-terminal.mp4

Require retrieval-evaluator

Set Shell "zsh"
Set FontSize 48
Set Width 3600
Set Height 1800
#Set LineHeight 1.2

Set Framerate 30
Set TypingSpeed 30ms
Set CursorBlink true
Set WaitTimeout 2m

Env HF_HUB_OFFLINE "1"
Env TRANSFORMERS_VERBOSITY "error"
Env STREAMLIT_BROWSER_GATHER_USAGE_STATS "false"
Env TOKENIZERS_PARALLELISM "false"

Hide

Type "PROMPT='%F{#5B56E1}>%f '; RPROMPT=''"
Enter

Type "autoload -Uz add-zle-hook-widget"
Enter
Type "demo_highlight() { region_highlight=(); [[ $BUFFER == retrieval-evaluator* ]] && region_highlight=('0 19 fg=#FAB387,bold'); }"
Enter
Type "add-zle-hook-widget line-pre-redraw demo_highlight"
Enter

# Use fresh local data so the recorded paths contain no machine-specific locations.
Type "cd $(mktemp -d /tmp/retrieval-evaluator-demo.XXXXXX)"
Enter
Type `python -c "import shutil; from importlib.resources import files; shutil.copytree(files('retrieval_evaluator') / 'data' / 'EXAMPLE', 'eval_data/EXAMPLE')"`
Enter
Sleep 500ms

# Remove all hidden setup from the visible terminal.
Type "clear"
Enter

Show

Sleep 800ms

Type "retrieval-evaluator init"
Sleep 500ms
Enter
Wait+Screen@10s /Created:/
Sleep 2000ms

Ctrl+L
Sleep 500ms

Type "retrieval-evaluator eval --collections EXAMPLE --model sentence-transformers/all-MiniLM-L6-v2"
Sleep 600ms
Enter
Wait+Screen@2m /MRR@10/
Sleep 5s

Ctrl+L
Sleep 500ms

Type "retrieval-evaluator dashboard --headless"
Sleep 500ms
Enter
Wait+Screen@30s /http:\/\/127.0.0.1:/
Sleep 3s

Hide
Ctrl+C
Sleep 500ms
