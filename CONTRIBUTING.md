# Contributing to SiegeScout

Contributions are welcome. Before starting substantial work, open an issue to
describe the problem or proposed change so that effort is not duplicated.
You can also join the [SiegeScout Discord server](https://discord.gg/VVM2UwdJQ4)
for questions, discussion, and contributor coordination.

## Workflow

1. Create a focused branch from `master`.
2. Keep changes small and explain behavior changes.
3. Add or update tests for logic changes.
4. Run `python -m pytest` before opening a pull request.
5. Include setup or documentation changes when user-facing behavior changes.

Do not commit credentials, private datasets, game captures containing personal
information, generated build output, or files listed in `.gitignore`.

## Improving computer vision and ML

CV contributions are especially welcome. The development pipeline is kept
separate from the shipped bot:

1. Capture representative game frames locally. Do not share captures that
   contain usernames, chat, or other personal information.
2. Label minimap points and screen objects with `eval/label_tool.py`. Prefer
   accurate, consistent labels over large noisy datasets. Record the source
   game/display conditions when they affect appearance.
3. Keep evaluation data separate from training data. Add labels and fixtures
   only when they are safe to redistribute and include the relevant license or
   permission.
4. Train or export a candidate model using the workflow in
   [training/README.md](training/README.md).
5. Run `python -m eval.score --detector all` against real labeled captures and
   compare the candidate with the heuristic baseline and the currently shipped
   model. Include per-class precision, recall, F1, inference time, input size,
   and hardware in the pull request.
6. Update the matching class order and input-size settings in `config.py` if
   the model contract changes. Do not enable a model by default until it
   improves the acceptance gate documented in the training guide and does not
   cause unacceptable CPU regressions.

Do not commit raw captures, dataset exports, training checkpoints, API keys, or
private Roboflow projects. A pull request may include a reviewed ONNX model in
`assets/models/` when its training data is redistributable, its size is
reasonable, and the model license is documented. Otherwise, contribute the
training code, labels, evaluation report, and instructions for reproducing the
model instead.

## Pull requests

Describe the motivation, the approach, and how you tested the change. Screenshots
are useful for GUI changes. Please be respectful and assume good intent.

Use the Discord server for general discussion, but keep bug reports and
reproducible technical issues in GitHub Issues so they remain searchable.
