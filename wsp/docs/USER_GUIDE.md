# WSP CameraTrap: user guide

WSP CameraTrap finds animals, people and vehicles in camera trap photos and
videos (MegaDetector) and identifies the species (SpeciesNet and WSP's own
models). Your images and results stay on your computer.

## 1. Install

1. Download `WSP-CameraTrap-Setup-<version>.exe` from the link you were sent
   (the app's GitHub releases page, or the WSP CameraTrap OneDrive folder).
2. Run it.
   - It installs for your user only and does not ask for admin rights.
   - If Windows shows "Windows protected your PC", click **More info ›
     Run anyway**.
3. Start **WSP CameraTrap** from the Start menu. The first start runs a
   one-time setup (about 10 minutes, needs the internet): it downloads the
   analysis environment and MegaDetector from the app's GitHub releases.
4. Install the models: download `WSP-CameraTrap-models.zip` from the link
   you were sent (OneDrive), then in the app choose **File › WSP model
   library… › Install models from a .zip…** and pick the file. It installs
   SpeciesNet, WSP's own models and the embedding model used for
   similarity search. Keep the .zip; you do not need to unpack it.

After that the app works offline.

## 2. Analyse a folder

1. **File › Analyse a folder…** (or *Analyse a folder* on the home page).
2. Pick the folder with your camera trap images or videos.
3. Choose the models: MegaDetector for detection, and SpeciesNet or a WSP
   model for species. For UK sites, set the country to **United Kingdom** so
   SpeciesNet does not suggest species that do not occur here.
4. Start. When it finishes you can check the results and save them: tables
   (CSV/XLSX), files sorted into folders per species, annotated copies, and
   more.

For long-term studies, create a **project** instead (**File › New
project…**): add sites and deployments, verify the AI results, and see
counts, activity patterns and maps.

## 3. Check the AI results

The AI makes mistakes. In the verify views you confirm or correct each
label (*Check labels*) and the number of animals (*Confirm counts*). Your
verification is stored in the project and is used in every chart and
export.

## 4. New WSP models

When a new WSP model is published you get a new `WSP-CameraTrap-models.zip`.
Install it the same way (**File › WSP model library… › Install models from
a .zip…**). Models you already have are kept.

## Troubleshooting

| Problem | What to do |
|---|---|
| "*… is not installed …*" | Install the model .zip: **File › WSP model library… › Install models from a .zip…**. |
| Setup stops while preparing the analysis environment | Check your internet connection and click **Retry**. |
| Anything else | **Help › Export diagnostic report**, and send it with a short description through **Help › Report a problem**. |

## Where things are

| What | Where |
|---|---|
| The app | `%LOCALAPPDATA%\Programs\WSP CameraTrap` |
| Your projects database, installed models, logs | `%USERPROFILE%\WSP-CameraTrap` |
| Analysis results of a folder run | a hidden `.wsp-cameratrap` folder inside the analysed folder, plus wherever you saved outputs |

Uninstalling the app (Windows Settings › Apps) asks whether to delete your
data folder as well. Your images are never touched.
