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
   one-time setup (about 10 minutes, needs the internet). It downloads:
   - the analysis environment, from the app's GitHub releases;
   - MegaDetector, SpeciesNet and WSP's own models, from the **WSP model
     library** on OneDrive. The link is built into the app; you do not need
     to set anything up.

After the setup the app works offline.

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

The app checks the WSP model library each time it starts. New models
appear in the model list with an **Install** button. To check right away,
open **File › WSP model library…** and click **Check for new models**.

## Troubleshooting

| Problem | What to do |
|---|---|
| "*… is not installed. No WSP model library …*" or "*The model library link …*" | Open **File › WSP model library…** and click **Check for new models**. If it still fails, the link may have expired: ask for the current link, paste it there and click **Connect**. |
| The library link asks you to sign in | The shared link must be set to *Anyone with the link*. Ask whoever publishes the models, or use **Use a folder…** with a synced copy of the library. |
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
