# Rehab MTL

This project is being built in verified stages. The current implementation covers
the KERAAL inventory and read-only Kinect XYZ loading. The model and training
components described in [the research plan](README_MTL_Rehabilitation_Plan.md)
are still pending.

## Current KERAAL files

- `data/keraal_manifest.csv`: one row per recording (841 rows) from group1A,
  group2A, and group3, with source paths relative to the KERAAL dataset root.
  Annotator labels are kept apart; group3 filename labels are stored as raw codes.
- `data/keraal_targets_preview.csv`: a review table of clip-level targets and
  masks. It preserves raw Group3 filename codes and adds a shared four-class
  `exercise_error_class` target, plus mapped global-evaluation and error-type
  targets for Group3.
- `data/keraal_subject_split_preview.csv`: a candidate participant-level split
  for review; all clips from each participant share one split. Group3 P1/P2/P3
  are assigned to train/validation/test respectively.
- `src/rehab_mtl/datasets/keraal.py`: strict Kinect frame parser and sample reader.
- `src/rehab_mtl/datasets/keraal_dataset.py`: PyTorch `KeraalMTLDataset`, batch collation (`collate_keraal_mtl`), and `build_keraal_dataloaders`.
- `scripts/validate_keraal.py`: checks the manifest, source-file joins, label
  values, temporal annotation states, and all Kinect frames.
- `scripts/build_keraal_targets_preview.py`: builds the target preview without
  changing the manifest or source dataset.
- `scripts/build_keraal_group3_csvs.py`: reads group3 Kinect and matching
  modality files, then updates the three CSVs. It preserves raw filename codes
  and writes mapped targets to the derived target preview.
- `scripts/preview_kinect_preprocessing.py`: shows a proposed coordinate offset
  for one clip in memory; it writes no transformed data.
- `scripts/validate_keraal_mtl_samples.py`: joins each Kinect sequence to its
  proposed targets by `sample_id` and checks masks across the dataset.
- `scripts/train_baseline.py`: end-to-end multi-task training runner using PyTorch DataLoaders.

The source dataset remains in its original location. The loader does not rewrite,
normalize, interpolate, or resample it. For v0 it reads the XYZ position values
from each joint and leaves out the quaternion values. Joint order follows the
KERAAL format README.

## Validate the local KERAAL data

From the project root, run:

```bash
python3 scripts/validate_keraal.py
```

The default dataset root is `~/Downloads/KERAAL`. Override it when the dataset
is stored elsewhere:

```bash
python3 scripts/validate_keraal.py --keraal-root /path/to/KERAAL
```

To rebuild the manifest, target preview, and subject split after adding group3:

```bash
python3 scripts/build_keraal_group3_csvs.py --keraal-root /path/to/KERAAL
```

Group3 has 540 recordings with Kinect, video, OpenPose, BlazePose, and Vicon
files, but no annotator XML files. Code `C` maps to `Correct`; the `E1`, `E2`,
and `E3` prefixes in codes such as `E1B1` map to `Error1`, `Error2`, and
`Error3`. The `B1`-`B3` suffix remains in the raw code and is not interpreted as
another target. Each derived target records its mapping in a resolution column.
Severity and body-part targets remain masked for Group3.

The validator is read-only. It fails with the sample ID and line number when a
Kinect frame does not contain exactly 175 finite numeric values, or when its
frame count differs from the manifest.

## Label handling

The loader preserves each annotator's original `Global Evaluation`, error type,
severity, and body-part labels. Blank values become `None` with a false label
mask. An explicit `none` error label stays labeled and has a true mask. A missing
temporal track remains unknown; temporal annotations are not converted into
frame targets because their timing has not been aligned to Kinect frames.
`load_keraal_sample` returns source XYZ by default. Its
`relative_to_initial_spine_base=True` option applies the reviewed-in-preview
translation in memory and records the origin used in the returned sample.

The pipe-separated label strings in the manifest are preserved as strings. A
training label mapper must split and interpret them according to a reviewed
ontology; it must not silently treat them as one class.

The current preview policy is provisional: if both annotators agree, their value
is used; if only one has a value, that value is shown; if both have different
values, that head is masked for that clip. A blank is not converted to `none`.
Exercise code comes from the manifest metadata. The preview also records review
flags for label combinations that need human interpretation. These rules must be
reviewed before the preview is used for training.

The shared `exercise_error_class` target uses `Correct`, `Error1`, `Error2`, and
`Error3`. Group3 codes map through their leading error prefix; Group1A/Group2A
use a resolved `Correct` global label or a resolved `Error1`-`Error3` label.
Incomplete, motionless, `Other`, unsupported, and disputed cases are masked for
this head. The four-class task follows the KERAAL experiment described in
[Marusic et al. (2025)](https://arxiv.org/abs/2504.13866). Raw labels and
per-task masks remain available for review.

Rebuild the preview with:

```bash
python3 scripts/build_keraal_targets_preview.py
```

Preview a coordinate offset for one clip with:

```bash
python3 scripts/preview_kinect_preprocessing.py --sample-id G1A-CTK-R1-Brest-038
```

Check Kinect-to-target joins for all samples with:

```bash
python3 scripts/validate_keraal_mtl_samples.py --relative-to-initial-spine-base
```

The validator reads one clip at a time. The target preview is still provisional;
this command checks data wiring and masks, not whether the annotation policy is
final.

## Next implementation steps

1. Review the target preview and finalize label and disagreement rules.
2. Make fixed subject-disjoint train/validation/test splits.
3. Add the PyTorch dataset, batching, shared backbone, and task heads, then
   compare multi-task training with single-task baselines.

See [README_MTL_Rehabilitation_Plan.md](README_MTL_Rehabilitation_Plan.md) for
the broader research plan and the roles of KERAAL, KIMORE, and other datasets.
