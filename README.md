# ssmprov
"SSM Improv".

This is meant to serve two purposes, first just to demonstrate saving and resuming of checkpoints with a state space model.

Secondly, to experiment with prompting using a set of output roles in a play-like turn based transcript - with the hope of building and forking possibly useful working specialist contexts for SSMs.

# Components:

* RWKV7 Runner - RWKV7.py

    TCP service for sampling and checkpointing.  Port 6502, null terminated post, response.
    Hard coded "\n\n"+"~~~(end)~~~"+"\n\n" turn stops and forced stop completion.

    Model .pth: https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/rwkv7-g0-7.2b-20250722-ctx4096.pth
    Convert to 8-bit gguf for runner.

* Composer GUI - COMPOSER.py

    Half and full-step turn prompting, Markdown+LaTeX rendering and raw transcript.

* tools.py

    Provides GET, PUT, QUOTE, RUN subcommands.

* install.sh

    Bash scripts to ~/bin, added to path if needed.
    Source to ~/src/ssmprov
    Creates working directory /_ and initial dot-files/paths within.

* Bash scripts:

    MODEL, COMPOSER - launch rwkv7 runner, composer gui in /_ using current Python environment.
    GET, PUT, QUOTE, RUN - Similar launchers for tools.py subcommands run within /_ and current environment.

# Composer GUI

The Composer GUI consists of a right-hand pane with a live display of .transcript.txt
and the left hand provides an interface for priming and single and half stepping
inference through the active runner.

Interface:

The top input box is for "!" prefix lines.  Any string in this box, if present, will be
prepended to the templated prompt sent.  "!loadname" presents the prompt to the specified
named checkpoint, "!loadname settings" presents the prompt to the checkpoint "loadname" with
the sampling settings "settings".  Finally, "!loadname settings savename" would present the
prompt to the checkpoint "loadname" with sampling settings "settings" and save the post-generation
checkpoint to "savename".

The main input area at the bottom is for free-form multi-line prompt input (shift+enter to send).

Above this are three drop-downs which may either be blank or allow selection (from left to right)
of: input role string, divider string, and output role string.

The values in these dropdowns are python expressions which evaluate to strings and are found in
.s1.txt, .s2.txt, and .s3.txt in the working directory (by default /_).

Additinally, '/' commands (such as /save, /load, etc.) may be sent to the runner as prompts,
without regard to dropdown settings.

Usage:

With a cold start, a primer text can be pasted to establish the end-marker usage and turn roles.
The file .counter should be pre-written at composer launch with the highest turn number in
the primer text, and the text should end with the start of an end marker, "\n\n~~~(", testing and
forcing completion on eval.

After this, the user may half-step and full-step through prompts depending on the divider and output
roles selected for input prompts: With the half-marker, the model will read in the input role turn
and simply finish the end marker to complete it.  With a full divider and specified output role name,
the model will (hopefully) generate role output and cleanly terminate with a full end marker on
completion.

