# Real-dataset validation summary

The dataset inventory was checked before making any physics-performance claim.

| Check | Result |
|---|---:|
| Files inspected | 115 |
| Files containing valid signal data | 114 |
| Valid signals matched to metadata | 114 |
| Metadata-ambiguous mappings | 8 |
| Unmatched files | 1 |
| Mapped records with tooth-count metadata | 0 of 114 |

The unmatched file is `3p5inch_stickout/c_1030_002.mat`.

Strict physics mode correctly refuses to continue when tooth count or unambiguous operating metadata is unavailable. The one-record demonstration in `../02_real_recording_demo/` uses the documented nonphysics research profile and is evidence that the implementation runs on a real waveform; it is not presented as a strict-physics validation of the entire dataset.

See `real_dataset_validation.json` for record-level evidence.
