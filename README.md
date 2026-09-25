# WSP CameraTrap

Camera trap image and video analysis for WSP ecologists: MegaDetector finds
animals, people and vehicles; SpeciesNet and WSP's own models identify the
species. Models are distributed through the WSP model library on OneDrive,
so nothing is downloaded from HuggingFace or Kaggle.

- **Ecologists:** [user guide](wsp/docs/USER_GUIDE.md)
- **Maintainer (models, releases):** [admin guide](wsp/docs/ADMIN_GUIDE.md)

## Repository layout

| Folder | What |
|---|---|
| `backend/` | FastAPI backend and ML workers |
| `frontend/` | React user interface |
| `electron/` | Desktop shell and Windows installer |
| `wsp/` | WSP additions: shipped model catalog, SpeciesNet and WSP model `inference.py`, the model library tool, docs |
| `training/` | WSP dataset preparation and classifier training |

Developer notes for the app itself are in [DEVELOPERS.md](DEVELOPERS.md).

## Credits and license

MIT license; third-party open-source notices are in [LICENSE](LICENSE). The
models have their own licenses: MegaDetector (MIT, Dan Morris), SpeciesNet
(Apache-2.0, Google).
