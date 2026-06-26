import zipfile
from pathlib import Path
import numpy as np


def zip_predictions(
    predictions_dir,
    output_zip,
    inner_folder="predictions",
    expected_count=946,
    validate=True,
):
    """
    Package .npy predictions into a challenge-ready submission zip.

    Layout inside the zip:
        <inner_folder>/<core_id>.npy   e.g. predictions/3001_BE_2023.npy

    Args:
        predictions_dir: folder containing the .npy prediction files.
        output_zip:      path of the .zip to write.
        inner_folder:    single top-level folder name required by the challenge.
        expected_count:  expected number of test patches (None to skip the check).
        validate:        if True, verify every file is a [4, H, W] float array.
    """
    predictions_dir = Path(predictions_dir)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(predictions_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {predictions_dir}")

    if validate:
        bad = []
        for f in files:
            arr = np.load(f, mmap_mode="r")  # mmap = cheap, no full load
            if arr.ndim != 3 or arr.shape[0] != 4:
                bad.append((f.name, tuple(arr.shape)))
        if bad:
            raise ValueError(f"{len(bad)} file(s) not [4, H, W], e.g. {bad[:3]}")

    if expected_count is not None and len(files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} predictions but found {len(files)}. "
            "Submission must cover all test patches."
        )

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f"{inner_folder}/{f.name}")

    size_mb = output_zip.stat().st_size / 1e6
    print(f"Wrote {output_zip}")
    print(f"  files zipped: {len(files)}")
    print(f"  sample entry: {inner_folder}/{files[0].name}")
    print(f"  zip size:     {size_mb:.1f} MB")
    return output_zip


zip_predictions(
    predictions_dir="/data/Geo_FM/emb2heights-baselines/baselines/My_TESSERA_ALPHA/predictions",
    output_zip="/data/Geo_FM/embed2heights/submission/submission_tessera_alpha_new.zip",
)