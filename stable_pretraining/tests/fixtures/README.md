# DAE-Net numerical reference

`dae_net_reference.py` contains the five network class definitions from
`czq142857/DAE-Net/model.py` at commit
`4f4af22519940b774ae47e9c8d9a99fbfd17ac9b`.

The upstream file SHA256 with CRLF line endings is
`46bbe3d9071c09c9a201f6e5bd9b5ab970be07581c0da99ad1e75adb77081064`.
The raw GitHub file with LF line endings has SHA256
`65bf2fe52cfc9fa0cdf65afdb4ba328bce977aca716fbe3d74db3c587c118a1e`.
Class definitions are unchanged; only the module header and imports differ.
Formatting is disabled for this fixture to preserve the original definitions.
The source is MIT licensed, Copyright (c) 2023 Zhiqin Chen; the full license is
included at `stable_pretraining/backbone/licenses/DAE_NET_LICENSE`.

Tests use this fixture offline and do not import the user's reference checkout
or require HDF5, OpenCV, scikit-image, ShapeNet data, pretrained weights, or CUDA.
The reference encoder forces float32, so end-to-end parity is tested in FP32;
the decoder is also compared in float64. Loss expressions are independently
transcribed from the same commit's training loop and complemented by
hand-calculated value and gradient checks.
