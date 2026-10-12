# Rehab MTL

This project is being built in verified stages. The current implementation covers
the KERAAL inventory and read-only Kinect XYZ loading. The model and training
components described in [the research plan](README_MTL_Rehabilitation_Plan.md)
are still pending.

## Current KERAAL files

- `data/keraal_manifest.csv`: one row per recording (301 rows), with source paths
  relative to the KERAAL dataset root and the two annotators' labels kept apart.
- `data/keraal_targets_preview.csv`: a review-only table of proposed clip-level
  targets and masks; it is derived from the manifest and is not a training set.
- `data/keraal_subject_split_preview.csv`: a candidate participant-level split
  for review; all clips from each participant share one split.
- `src/rehab_mtl/datasets/keraal.py`: strict Kinect frame parser and sample reader.
- `scripts/validate_keraal.py`: checks the manifest, source-file joins, label
  values, temporal annotation states, and all Kinect frames.
- `scripts/build_keraal_targets_preview.py`: builds the target preview without
  changing the manifest or source dataset.
- `scripts/preview_kinect_preprocessing.py`: shows a proposed coordinate offset
  for one clip in memory; it writes no transformed data.
- `scripts/validate_keraal_mtl_samples.py`: joins each Kinect sequence to its
  proposed targets by `sample_id` and checks masks across the dataset.

The source dataset remains in its original location. The loader does not rewrite,
normalize, interpolate, or resample it. For v0 it reads the XYZ position values
from each joint and leaves out the quaternion values. Joint order follows the
KERAAL format README.

## Validate the local KERAAL data

From the project root, run:

```bash
python3 scripts/validate_keraal.py
```

The default dataset root is `~/Documents/KERAAL`. Override it when the dataset is
stored elsewhere:

```bash
python3 scripts/validate_keraal.py --keraal-root /path/to/KERAAL
```

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
