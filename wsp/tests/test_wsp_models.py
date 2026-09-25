"""WSP model library tool and the inference.py files it publishes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "wsp" / "tools"))
sys.path.insert(0, str(REPO / "training"))

import wsp_library  # noqa: E402

LABELS = "\n".join([
    "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
    "aaaaaaaa-1;mammalia;carnivora;mustelidae;meles;meles;eurasian badger",
    "bbbbbbbb-2;mammalia;carnivora;mustelidae;lutra;lutra;eurasian otter",
    "cccccccc-3;mammalia;carnivora;mustelidae;;;",
    "dddddddd-4;mammalia;carnivora;mustelidae;lutra;;eurasian otter",
    "eeeeeeee-5;aves;;;;;",
])


def _load_inference(path: Path):
    spec = importlib.util.spec_from_file_location(f"inf_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speciesnet_names_follow_the_app_dedup_rule(tmp_path):
    labels = tmp_path / "x.labels.txt"
    labels.write_text(LABELS)
    inference = _load_inference(wsp_library.SPECIESNET_INFERENCE)
    names = inference.class_names_from_labels(labels)
    assert names == [
        "blank", "eurasian badger", "eurasian otter", "mustelidae", "lutra", "aves",
    ]
    # taxonomy.csv written by the tool uses the very same names.
    rows = wsp_library.speciesnet_taxonomy_rows(labels)
    assert [r["model_class"] for r in rows] == names


def test_speciesnet_reads_the_labels_file_the_geofence_reads(tmp_path):
    """With a dated labels file next to the original, the classifier, the
    taxonomy.csv and the app's geofence (find_labels_file: the first of the
    sorted matches) must all read the same one."""
    sn = tmp_path / "speciesnet"
    sn.mkdir()
    (sn / "always_crop_99710272_22x8_v12_epoch_00148.pt").write_bytes(b"w")
    (sn / "x.labels.txt").write_text("u1;;;;;;original\n")
    (sn / "x.labels.20251208.txt").write_text("u1;;;;;;dated\n")
    geofence_pick = sorted(sn.glob("*.labels*.txt"))[0]
    assert geofence_pick.name == "x.labels.20251208.txt"

    inference = _load_inference(wsp_library.SPECIESNET_INFERENCE)
    assert inference.ModelInference(sn, sn / "unused.pt").names == ["dated"]

    lib = tmp_path / "lib"
    assert wsp_library.main(["add-speciesnet", str(lib), str(sn)]) == 0
    taxonomy = (lib / "cls" / "SPECIESNET-v4-0-2-A" / "taxonomy.csv").read_text()
    assert "dated" in taxonomy and "original" not in taxonomy


def test_library_init_md_and_speciesnet(tmp_path):
    lib = tmp_path / "WSP CameraTrap" / "models"
    assert wsp_library.main(["init", str(lib)]) == 0

    md = tmp_path / "md_v5a.0.0.pt"
    md.write_bytes(b"md")
    assert wsp_library.main(["add-md", str(lib), str(md)]) == 0

    sn = tmp_path / "speciesnet"
    sn.mkdir()
    (sn / "always_crop_99710272_22x8_v12_epoch_00148.pt").write_bytes(b"w")
    (sn / "always_crop_99710272_22x8_v12_epoch_00148.labels.txt").write_text(LABELS)
    (sn / "geofence_release.2025.02.27.0702.json").write_text("{}")
    assert wsp_library.main(["add-speciesnet", str(lib), str(sn)]) == 0

    catalog = json.loads((lib / "models.json").read_text())
    assert [m["model_id"] for m in catalog["models"]["det"]] == ["MD5A-0-0"]
    assert [m["model_id"] for m in catalog["models"]["cls"]] == ["SPECIESNET-v4-0-2-A"]
    folder = lib / "cls" / "SPECIESNET-v4-0-2-A"
    assert (folder / "inference.py").is_file()
    assert (folder / "taxonomy.csv").read_text().startswith("model_class,class,order")
    assert (lib / "det" / "MD5A-0-0" / "md_v5a.0.0.pt").read_bytes() == b"md"


def test_speciesnet_inference_runs_on_a_crop(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx2torch")

    class Tiny(torch.nn.Module):
        def forward(self, x):  # x: (N, 480, 480, 3) in [0, 1]
            m = x.mean(dim=(1, 2))
            return torch.cat([m, m[:, :3]], dim=1)  # 6 logits

    folder = tmp_path / "SPECIESNET-v4-0-2-A"
    folder.mkdir()
    (folder / "sn.labels.txt").write_text(LABELS)
    weights = folder / "sn.pt"
    # Like the real SpeciesNet, a pickled torch.fx GraphModule.
    torch.save(torch.fx.symbolic_trace(Tiny()), weights)

    module = _load_inference(wsp_library.SPECIESNET_INFERENCE)
    model = module.ModelInference(folder, weights)
    model.load_model()
    image = Image.new("RGB", (640, 480), (200, 100, 50))
    crop = model.get_crop(image, (0.25, 0.25, 0.5, 0.5))
    assert crop.size == (320, 240)
    tensor = model.get_tensor(crop)
    assert tensor.shape == (480, 480, 3) and 0.0 <= tensor.min() <= tensor.max() <= 1.0
    result = model.get_classification(crop)
    assert [name for name, _ in result] == module.class_names_from_labels(folder / "sn.labels.txt")
    assert abs(sum(p for _, p in result) - 1.0) < 1e-5
    assert model.get_class_names()["1"] == "blank"


def test_published_wsp_model_loads_and_classifies(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from wildlife_classifier import build_classifier_model

    names = ["badger", "otter", "other"]
    net = build_classifier_model("mobilenet_v3_small", len(names))
    checkpoint = tmp_path / "wsp_uk_v1.pth"
    torch.save({
        "state_dict": net.state_dict(),
        "class_names": names,
        "input_size": 224,
        "arch": "mobilenet_v3_small",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }, checkpoint)

    lib = tmp_path / "lib"
    taxonomy = tmp_path / "tax.csv"
    taxonomy.write_text(
        "model_class,class,order,family,genus,species\n"
        "badger,mammalia,carnivora,mustelidae,meles,meles\n"
    )
    assert wsp_library.main([
        "add-model", str(lib), "--checkpoint", str(checkpoint),
        "--id", "WSP-UK-v1", "--taxonomy", str(taxonomy),
    ]) == 0
    # A second publish under the same id is refused.
    assert wsp_library.main([
        "add-model", str(lib), "--checkpoint", str(checkpoint), "--id", "WSP-UK-v1",
    ]) == 1

    folder = lib / "cls" / "WSP-UK-v1"
    entry = json.loads((lib / "models.json").read_text())["models"]["cls"][0]
    assert entry["model_fname"] == "model.pt" and entry["species_list"] == names
    assert "badger,mammalia,carnivora" in (folder / "taxonomy.csv").read_text()

    module = _load_inference(folder / "inference.py")
    model = module.ModelInference(folder, folder / "model.pt")
    assert model.get_class_names() == {"1": "badger", "2": "otter", "3": "other"}
    model.load_model()
    crop = model.get_crop(Image.new("RGB", (800, 600)), (0.1, 0.1, 0.3, 0.3))
    batch = [model.get_tensor(crop), model.get_tensor(crop)]
    import numpy as np

    results = model.classify_batch(np.stack(batch))
    assert len(results) == 2
    assert [n for n, _ in results[0]] == names
