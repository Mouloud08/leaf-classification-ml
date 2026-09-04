# Dataset provenance

This project uses `raw/train.csv` from Kaggle's
[Leaf Classification competition](https://www.kaggle.com/competitions/leaf-classification/data).
The file is retained in this portfolio repository for reproducibility.

## Verified file

| Property | Value |
| --- | --- |
| Relative path | `data/raw/train.csv` |
| Observations | 990 labelled leaf specimens |
| Target classes | 99 species |
| Predictive features | 192 numeric descriptors |
| Feature groups | 64 margin, 64 shape, 64 texture |
| Additional columns | `id`, `species` |
| Total columns | 194 |
| File size | 1,568,525 bytes |
| SHA-256 | `8A89EAE52B86999E720916C6145B0735D9BC830E3358D45AE8D3C195C86F50DC` |

Verify the checked file from the repository root:

```powershell
Get-FileHash -Algorithm SHA256 data\raw\train.csv
```

On macOS or Linux:

```bash
sha256sum data/raw/train.csv
```

The expected digest is the SHA-256 value in the table above. A different value
means the data does not match the file used for the reported experiments.

## Schema

- `id`: specimen identifier; excluded from model features.
- `species`: multiclass prediction target.
- `margin1`–`margin64`: leaf-margin descriptors.
- `shape1`–`shape64`: leaf-shape descriptors.
- `texture1`–`texture64`: leaf-texture descriptors.

## Usage and rights

The data is third-party material and is not covered by this repository's
project copyright notice. Review Kaggle's
[competition rules](https://www.kaggle.com/competitions/leaf-classification/rules)
before using or redistributing it. No ownership of the dataset is claimed here.
