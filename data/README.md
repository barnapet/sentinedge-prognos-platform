# Dataset: NASA IMS (Rexnord) Bearing Dataset

This directory is **gitignored** — the raw dataset is not committed to the repository (see
root `.gitignore`). This README documents where it comes from and how to reproduce it.

## Source

- **Name:** IMS Bearing Dataset (Center for Intelligent Maintenance Systems, University of
  Cincinnati), distributed via the NASA Prognostics Center of Excellence (PCoE) Data Set
  Repository.
- **Download URL used:** https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
  (~1.1 GB, official NASA-hosted mirror; this is the same dataset referenced from the PCoE
  data repository page and the common Kaggle mirrors).
- **License:** public domain / NASA data usage terms — see the PDF inside the archive
  (`Readme Document for IMS Bearing Data.pdf`) for full attribution details.

## How to re-download

```bash
cd data/
curl -L -o nasa_bearings.zip https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
unzip nasa_bearings.zip                     # -> "4. Bearings/IMS.7z"
```

The zip contains a single 7z archive (`IMS.7z`), which in turn contains three RAR archives
(one per experiment). Extraction requires `7z`/`py7zr` (for the outer 7z) and `unrar` (for the
inner per-experiment RAR files) — neither ships by default on a minimal system.

```bash
# extract the 7z (py7zr, since 7z/p7zip may not be installed):
python3 -m venv /tmp/venv7z && /tmp/venv7z/bin/pip install py7zr
/tmp/venv7z/bin/python -m py7zr x "4. Bearings/IMS.7z" ./IMS_extracted

# extract the three inner RAR archives:
cd IMS_extracted
unrar x -y 1st_test.rar
unrar x -y 2nd_test.rar
unrar x -y 3rd_test.rar
```

**Known archive quirks** (present in the original NASA distribution, not introduced here) —
corrected against the actual extraction performed by `.github/workflows/notebook-ci.yml`, which
downloads and lays out the dataset from scratch on every PR:
- `1st_test.rar` and `2nd_test.rar` each extract to a single folder named after the experiment
  (`IMS_extracted/1st_test/`, `IMS_extracted/2nd_test/`) containing the snapshot files
  directly — no redundant second level.
- `3rd_test.rar` internally names its top-level folder **`4th_test`**, and nests the actual data
  one level deeper under a `txt/` subfolder — so the data lands in `IMS_extracted/4th_test/txt/`,
  with no `3rd_test/` component in the path at all.

After extraction, flatten so the final layout matches [Structure](#structure) below, then discard
the intermediate `nasa_bearings.zip`, `4. Bearings/`, and `IMS_extracted/` scaffolding:

```bash
cd ..                       # back to data/
mkdir -p raw
mv IMS_extracted/1st_test    raw/1st_test
mv IMS_extracted/2nd_test    raw/2nd_test
mv IMS_extracted/4th_test/txt raw/3rd_test     # note: 4th_test/txt, not 3rd_test/
rm -rf nasa_bearings.zip "4. Bearings" IMS_extracted
```

(These are the same moves the CI workflow performs; if the upstream archive layout ever changes,
the CI's `find data/IMS_extracted -maxdepth 4 -type d` debug step prints the structure it
actually got.)

## Structure

```
data/
└── raw/
    ├── 1st_test/    2,156 files   2003-10-22 → 2003-11-25   8 channels/file (2 per bearing, 4 bearings)
    ├── 2nd_test/      984 files   2004-02-12 → 2004-02-19   4 channels/file (1 per bearing, 4 bearings)
    ├── 3rd_test/    6,324 files   2004-03-04 → 2004-04-18   4 channels/file (1 per bearing, 4 bearings)
    └── Readme Document for IMS Bearing Data.pdf
```

Each filename is a timestamp (`YYYY.MM.DD.HH.MM.SS`) marking when that 1-second snapshot was
recorded, at roughly 10-minute intervals across the run-to-failure life of the bearing set.

## Validation performed (Issue 1, M1-EDA)

Verified against `docs/PRD.md` Section 6 across all 9,464 files:

- [x] Three independent experiments present (`1st_test`, `2nd_test`, `3rd_test`).
- [x] Every file has exactly 20,480 data points (rows) — a 1-second snapshot at 20 kHz.
- [x] `1st_test` has 8 channels (columns) per file, matching Set 1's 2-sensors-per-bearing
      config; `2nd_test` and `3rd_test` have 4 channels each, matching 1-sensor-per-bearing.

No content/label validation (failure mode, health-state thresholds) has been done yet — that's
downstream EDA work, not part of this download/validate issue.
