import pytest
from embeddings.dataloaders import _get_file_pairs

def test_get_file_pairs_correct_matches_and_order(tmp_path):
    data_dir = tmp_path / "data"
    label_dir = tmp_path / "label"
    data_dir.mkdir()
    label_dir.mkdir()

    # Create dummy files
    (data_dir / "gee_emb_2013_OG.tif").touch()
    (data_dir / "gee_emb_0003_BE.tif").touch()
    
    (label_dir / "label_2013_OG_2023.tif").touch()
    (label_dir / "label_0003_BE_2023.tif").touch()

    pairs = _get_file_pairs(data_dir, label_dir)

    assert len(pairs) == 2
    
    # Check order (0003 should be first because of alphanumeric sorting)
    assert pairs[0][0].name == "gee_emb_0003_BE.tif"
    assert pairs[0][1].name == "label_0003_BE_2023.tif"
    
    assert pairs[1][0].name == "gee_emb_2013_OG.tif"
    assert pairs[1][1].name == "label_2013_OG_2023.tif"

def test_get_file_pairs_no_match_raises_value_error(tmp_path):
    data_dir = tmp_path / "data"
    label_dir = tmp_path / "label"
    data_dir.mkdir()
    label_dir.mkdir()

    # Unequal number of files will raise ValueError because of strict=True in zip
    (data_dir / "gee_emb_2013_OG.tif").touch()
    (data_dir / "gee_emb_0003_BE.tif").touch()
    
    (label_dir / "label_2013_OG_2023.tif").touch()

    with pytest.raises(ValueError):
        _get_file_pairs(data_dir, label_dir)

def test_get_file_pairs_empty_dirs(tmp_path):
    data_dir = tmp_path / "data"
    label_dir = tmp_path / "label"
    data_dir.mkdir()
    label_dir.mkdir()

    pairs = _get_file_pairs(data_dir, label_dir)
    assert pairs == []
