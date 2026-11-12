import pathlib

from retrieval_evaluator.cli import main


def test_init_creates_config_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init"])

    assert (tmp_path / "config" / "retrieval-evaluator.yaml").exists()
    assert (tmp_path / "config" / "embedders.yml").exists()


def test_init_does_not_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config" / "retrieval-evaluator.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("original content")

    main(["init"])

    assert config_path.read_text() == "original content"


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config" / "retrieval-evaluator.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("original content")

    main(["init", "--force"])

    assert config_path.read_text() != "original content"
    assert "source" in config_path.read_text()


def test_init_global_writes_to_home(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.chdir(tmp_path)
    main(["init", "--global"])

    global_dir = tmp_path / ".config" / "retrieval-evaluator"
    config_path = global_dir / "retrieval-evaluator.yaml"
    embedders_path = global_dir / "embedders.yml"
    assert config_path.exists()
    assert embedders_path.exists()
    assert str(embedders_path) in config_path.read_text()
    assert not (tmp_path / "config").exists()
