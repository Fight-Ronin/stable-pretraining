# Third-party components

## DAE-Net

`stable_pretraining/backbone/dae_net.py` and `stable_pretraining/losses/dae_net.py`
adapt the network, branch initialization, and objective from
[czq142857/DAE-Net](https://github.com/czq142857/DAE-Net), commit
`4f4af22519940b774ae47e9c8d9a99fbfd17ac9b` (MIT, Copyright 2023 Zhiqin Chen).
The complete license is in `stable_pretraining/backbone/licenses/DAE_NET_LICENSE`
and is included in built distributions.

The port adds dictionary inputs/outputs, separate encode/decode/template queries,
input validation, and model-dtype voxel conversion. Construction initializes
branches as at the start of upstream training. Network parameter names are
preserved. The test fixture contains the unmodified upstream network class
definitions with dependency-only imports so numerical tests run offline.

`stable_pretraining/utils/part_segmentation_evaluation.py` includes a count-based
compatibility implementation of the same commit's semantic branch mapping
search. The compatibility result separately exposes the reference
search's tracked score and the score recomputed from its final mapping.
The same module implements the release's nearest-confident-point label
propagation. `stable_pretraining/data/_dae_net/segmentation.py` uses its surface
coordinate transform, with manifest identity, hash and label validation added.

## Jet

`stable_pretraining/backbone/jet.py` adapts the transformer conditioner and
affine-coupling architecture from [btrude/jet-pytorch](https://github.com/btrude/jet-pytorch),
commit `d71d0dcfdc8b190f9b01795fd188742ce669130d`, distributed under Apache-2.0.
The full license is in `stable_pretraining/backbone/licenses/JET_LICENSE` and is
included in built distributions. The rest of stable-pretraining retains its MIT license.

Changes include BCHW input, indexed permutations, a configurable exponential
scale with a lower floor, FP32-or-higher affine arithmetic and log determinants,
diagnostics, activation checkpointing, and checkpoint configuration validation.
This port uses its own state-dict layout; existing upstream and deepstats
checkpoints require an explicit conversion and cannot be loaded directly.
