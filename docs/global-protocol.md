# Codrawing global and replay protocol

The live viewer opens `/global` and receives a full public state snapshot after each resolved turn. A snapshot contains the target, complete canvas, public messages from that turn, accepted slots, collision slots, progress counters, and the shared image-model target score and predictions for that canvas. The viewer displays the score, turn-to-turn delta, best target label, and its rank.

In replay mode, the game starts with `COGAME_LOAD_REPLAY_URI`. The replay viewer opens `/replay` and receives one payload whose `frames` array contains those same public snapshots, including the score at every turn. The default browser viewer autoplays, loops, supports seeking, and can download the displayed canvas as a scaled PNG.
