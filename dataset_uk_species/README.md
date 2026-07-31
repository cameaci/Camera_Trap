# UK Species Dataset Contract

Use this folder for the UK-focused classifier dataset used by `train_species_classifier.py`.

## Required structure

```text
dataset_uk_species/
  train/
    badger/
    otter/
    other/
  val/
    badger/
    otter/
    other/
  test/
    badger/
    otter/
    other/
```

## Baseline minimum images per class

- `train`: 400
- `val`: 100
- `test`: 100

These minimums are enforced by default in both:

- `validate_species_dataset.py`
- `train_species_classifier.py` (unless `--skip-dataset-validation` is used)

## Class definitions

- `badger`: UK badger detections/crops.
- `otter`: UK otter detections/crops.
- `other`: non-target UK classes (for example fox, deer, hare, hedgehog, birds, livestock, people, false crops).

## Data quality requirements

- Keep only licensing-permitted data for model training.
- Remove exact duplicates across `train/val/test`.
- Remove low-quality, unreadable, or corrupt images.
- Keep the same class semantics across all splits.

## Validation command

```powershell
python validate_species_dataset.py --data-root dataset_uk_species --json-output reports/dataset_validation.json
```

## Open-data fetch command (UK + iNaturalist)

```powershell
python fetch_uk_open_data.py --out dataset_uk_species --licenses cc0,cc-by,cc-by-nc --uk-place-id 6857 --targets 600,150,150 --reset
```

This command writes:

- images to `dataset_uk_species/{train,val,test}/{badger,otter,other}/`
- metadata to `dataset_uk_species/manifest.csv`

Manifest columns:

- `class_name,source,observation_id,photo_id,license,url,sha256,split`

## Training command

```powershell
python train_species_classifier.py --data-root dataset_uk_species --arch mobilenet_v3_small --epochs 30 --batch-size 32 --lr 1e-3 --output wildlife_model_badger_otter.pth
```
