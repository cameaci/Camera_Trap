# WSP CameraTrap: user guide

WSP CameraTrap finds animals, people and vehicles in camera trap photos and
videos (MegaDetector) and identifies the species (SpeciesNet and WSP's own
models). Your images and results stay on your computer.

## 1. Get the app and the models

Everything comes from the shared **WSP CameraTrap** folder on OneDrive
(SharePoint). You need to do this only once.

1. Open the SharePoint link you were sent for **WSP CameraTrap** and click
   **Add shortcut to My files**. OneDrive now syncs the folder to your
   laptop. Its `models` folder is the **WSP model library**.
2. In File Explorer, open `OneDrive - WSP › WSP CameraTrap › installer` and
   run `WSP-CameraTrap-Setup-<version>.exe`.
   - It installs for your user only and does not ask for admin rights.
   - If Windows shows "Windows protected your PC", click **More info ›
     Run anyway**.
3. Start **WSP CameraTrap** from the Start menu. The first start runs a
   one-time setup (10 to 20 minutes, needs the internet): it prepares the
   analysis environment and installs MegaDetector and SpeciesNet from the
   model library.

The app finds the synced library by itself. If it does not (for example
the library is on a network drive), use **File › WSP model library…** and
choose the `WSP CameraTrap/models` folder.

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

When a new WSP model is published to the library, the app shows it in the
model list with an **Install** button. Installing copies it from OneDrive
to your laptop; it keeps working offline afterwards.

## Troubleshooting

| Problem | What to do |
|---|---|
| "*… is not installed. No WSP model library folder is set up …*" | Sync the WSP CameraTrap folder (step 1.1), wait until OneDrive shows it as up to date, then **File › WSP model library…**. |
| "*The saved folder cannot be reached*" | OneDrive is not running or the network drive is not connected. Start OneDrive, or choose the folder again. |
| Setup stops while preparing the analysis environment | Check your internet connection and click **Retry**. The setup downloads Python packages once; afterwards the app works offline. |
| The OneDrive folder shows cloud icons | Right-click the `WSP CameraTrap/models` folder › **Always keep on this device**, so the models are downloaded before you install them. |
| Anything else | **Help › Export diagnostic report**, and send it with a short description through **Help › Report a problem**. |

## Where things are

| What | Where |
|---|---|
| The app | `%LOCALAPPDATA%\Programs\WSP CameraTrap` |
| Your projects database, installed models, logs | `%USERPROFILE%\WSP-CameraTrap` |
| Analysis results of a folder run | a hidden `.wsp-cameratrap` folder inside the analysed folder, plus wherever you saved outputs |

Uninstalling the app (Windows Settings › Apps) asks whether to delete your
data folder as well. Your images are never touched.
