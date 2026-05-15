import pytest
from embeddings.dataloaders import _get_file_pairs

def test_get_file_pairs_correct_matches_and_order(tmp_path):
    data_dir = tmp_path / "data"
    label_dir = tmp_path / "label"
    data_dir.mkdir()
    label_dir.mkdir()

    # Create dummy files
    (data_dir / "gee_emb_2013_OG.npy").touch()
    (data_dir / "gee_emb_0003_BE.npy").touch()
    
    (label_dir / "label_2013_OG_2023.npy").touch()
    (label_dir / "label_0003_BE_2023.npy").touch()

    pairs = _get_file_pairs(data_dir, label_dir)

    assert len(pairs) == 2
    
    # Check order (0003 should be first because of alphanumeric sorting)
    assert pairs[0][0].name == "gee_emb_0003_BE.npy"
    assert pairs[0][1].name == "label_0003_BE_2023.npy"
    
    assert pairs[1][0].name == "gee_emb_2013_OG.npy"
    assert pairs[1][1].name == "label_2013_OG_2023.npy"

def test_get_file_pairs_no_match_raises_value_error(tmp_path):
    data_dir = tmp_path / "data"
    label_dir = tmp_path / "label"
    data_dir.mkdir()
    label_dir.mkdir()

    # Unequal number of files will raise ValueError because of strict=True in zip
    (data_dir / "gee_emb_2013_OG.npy").touch()
    (data_dir / "gee_emb_0003_BE.npy").touch()
    
    (label_dir / "label_2013_OG_2023.npy").touch()

    with pytest.raises(ValueError):
        _get_file_pairs(data_dir, label_dir)

def test_get_file_pairs_empty_dirs(tmp_path):
    data_dir = tmp_path / "data"
    label_dir = tmp_path / "label"
    data_dir.mkdir()
    label_dir.mkdir()

    pairs = _get_file_pairs(data_dir, label_dir)
    assert pairs == []
