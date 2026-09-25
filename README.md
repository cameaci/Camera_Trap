# WSP CameraTrap

Camera trap image and video analysis for WSP ecologists: MegaDetector finds
animals, people and vehicles; SpeciesNet and WSP's own models identify the
species. Models are distributed through the WSP model library on OneDrive,
so nothing is downloaded from HuggingFace or Kaggle.

- **Ecologists:** [user guide](wsp/docs/USER_GUIDE.md)
- **Maintainer (models, releases):** [admin guide](wsp/docs/ADMIN_GUIDE.md)
- **Design and plan:** [wsp/docs/WSP_PLAN.md](wsp/docs/WSP_PLAN.md)

## Repository layout

| Folder | What |
|---|---|
| `backend/` | FastAPI backend and ML workers (from AddaxAI) |
| `frontend/` | React user interface (from AddaxAI) |
| `electron/` | Desktop shell and Windows installer |
| `wsp/` | WSP additions: shipped model catalog, SpeciesNet and WSP model `inference.py`, the model library tool, docs |
| `training/` | WSP dataset preparation and classifier training |

Developer notes for the app itself are in [DEVELOPERS.md](DEVELOPERS.md).

## Credits and license

WSP CameraTrap is built on [AddaxAI](https://github.com/PetervanLunteren/AddaxAI)
by Peter van Lunteren (Addax Data Science), MIT license. The models have
their own licenses: MegaDetector (MIT, Dan Morris), SpeciesNet (Apache-2.0,
Google). See [LICENSE](LICENSE).
