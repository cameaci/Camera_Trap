# WSP CameraTrap: yönetici kılavuzu

Bu kılavuz uygulamayı yayınlayan ve modelleri dağıtan kişi içindir (sen).
Ekolog arkadaşların için olan kılavuz `USER_GUIDE.md`.

## Genel yapı

```
GitHub (bu repo)                     OneDrive / SharePoint: "WSP CameraTrap"
  ├─ kaynak kod                        ├─ installer/WSP-CameraTrap-Setup-<ver>.exe
  ├─ CI: testler                       └─ models/                ← "WSP model library"
  └─ Release → Windows installer           ├─ models.json        (katalog)
                                           ├─ det/MD5A-0-0/md_v5a.0.0.pt
                                           ├─ cls/SPECIESNET-v4-0-2-A/…
                                           └─ cls/WSP-UK-v1/…
```

Uygulama modelleri şu sırayla arar:

1. Kullanıcının yerel klasörü `%USERPROFILE%\WSP-CameraTrap\models`
2. OneDrive'daki model kütüphanesi (otomatik bulunur ya da *File › WSP
   model library…* ile seçilir)
3. Yalnızca MegaDetector için GitHub Releases'taki resmi indirme adresi

HuggingFace'e ve Kaggle'a hiç bağlanmaz.

## 1. OneDrive kütüphanesini bir kez kurmak

Kütüphaneyi herkesin **okuyabildiği**, yalnızca senin **yazabildiğin** bir
SharePoint klasöründe tut (örneğin ekip sitesinde `WSP CameraTrap`).

1. Klasörü kendi bilgisayarına senkronize et ve kütüphaneyi hazırla:

   ```bash
   LIB="C:/Users/<sen>/OneDrive - WSP/WSP CameraTrap/models"
   python wsp/tools/wsp_library.py init "$LIB"
   ```

2. **MegaDetector v5a**: `md_v5a.0.0.pt` dosyasını GitHub'dan indir
   (https://github.com/agentmorris/MegaDetector/releases/tag/v5.0) ve ekle:

   ```bash
   python wsp/tools/wsp_library.py add-md "$LIB" md_v5a.0.0.pt
   ```

3. **SpeciesNet v4.0.2a**: Kaggle ve HuggingFace şirket ağında engelli
   olduğu için bu dosyaları **bir kez şirket ağı dışında** indir: Kaggle'da
   `google/speciesnet`, *PyTorch*, sürüm **v4.0.2a**. Klasörde
   `always_crop_99710272_22x8_v12_epoch_00148.pt`, `…labels.txt`,
   `geofence_release…json`, `info.json` ve `taxonomy_release.txt` bulunur.
   Ardından:

   ```bash
   python wsp/tools/wsp_library.py add-speciesnet "$LIB" path/to/speciesnet-v4.0.2a
   ```

   Bu komut dosyaları kopyalar, `inference.py` dosyasını ekler, `taxonomy.csv`
   dosyasını üretir ve katalog girdisini yazar. SpeciesNet Apache-2.0,
   MegaDetector MIT lisanslıdır; ikisi de şirket içinde dağıtılabilir.

4. Klasörün SharePoint linkini ekiple paylaş. Arkadaşların *Add shortcut to
   My files* yapınca uygulama kütüphaneyi kendisi bulur.

## 2. Kendi modelini yayınlamak (aşamalı dağıtım)

1. Modeli eğit (`training/` altında, bkz. `training/train_species_classifier.py`).
   Çıktı olarak `state_dict`, `class_names`, `arch`, `input_size` ve
   `normalization` içeren bir `.pth` dosyası oluşur.
2. Kütüphaneye ekle. **Her yeni sürüm için yeni bir id kullan** (`WSP-UK-v1`,
   `WSP-UK-v2`, …). Böylece eski sürümü kullananlar bozulmaz, yeni sürüm
   listede ayrı bir model olarak görünür.

   ```bash
   python wsp/tools/wsp_library.py add-model "$LIB" \
       --checkpoint training/wsp_uk_v2.pth \
       --id WSP-UK-v2 --name "WSP UK mammals v2" \
       --taxonomy uk_taxonomy.csv        # isteğe bağlı: model_class,class,order,family,genus,species
   ```

3. OneDrive senkronize edince arkadaşların uygulamayı bir sonraki açışlarında
   (ya da *File › WSP model library…* ekranına bakınca) yeni modeli görür ve
   **Install** ile kurar.

`inference.py`'yi veya `taxonomy.csv`'yi aynı id altında düzeltirsen
uygulama bunu "update available" olarak gösterir ve yalnızca değişen
dosyaları kopyalar. Ağırlık dosyasını aynı id altında değiştirme; bunun
yerine yeni bir id yayınla.

Şu an paketleme aracı eğitim scriptinin `mobilenet_v3_small` mimarisini
destekliyor. SpeciesNet omurgası üzerine fine-tune edeceğin modeller için
`wsp/models/SPECIESNET-v4-0-2-A/inference.py` başlangıç noktasıdır: aynı
ön işleme, farklı ağırlık ve etiket dosyası.

## 3. Uygulamanın yeni sürümünü çıkarmak

1. Değişiklikler `main` dalında olsun ve CI yeşil olsun.
2. GitHub'da **Releases › Draft a new release**: tag `v1.0.0` gibi bir değer
   olsun, notları yaz, **Publish**.
3. `Build Windows installer` workflow'u installer'ı üretir ve release'e
   ekler (yaklaşık 30 dakika sürer).
4. `WSP-CameraTrap-Setup-1.0.0.exe` dosyasını OneDrive'daki `installer/`
   klasörüne koy. Eski sürümü silebilirsin.

İmza: WSP IT'den bir code-signing sertifikası (`.pfx`) alırsan, repo
ayarlarında `WSP_PFX_BASE64` ve `WSP_PFX_PASSWORD` secret'larını tanımla.
Bundan sonra installer imzalı çıkar ve SmartScreen uyarısı kaybolur.

## 4. AddaxAI'daki güncellemeleri almak

Bu repo AddaxAI'ın ince bir fork'u. WSP değişiklikleri `# WSP` yorumlarıyla
işaretli ve az sayıda dosyada:

- `backend/app/ml/model_library.py`
- `backend/app/api/routers/wsp.py`
- `frontend/src/lib/wsp.ts`
- `frontend/src/api/wsp.ts`
- `wsp/`

```bash
git remote add upstream https://github.com/PetervanLunteren/AddaxAI.git   # bir kez
git fetch upstream
git checkout -b upstream-sync
git merge upstream/main          # çakışmaları çöz; WSP satırlarını koru
# testler: backend (pytest), frontend (npm run typecheck), wsp/tests
```

Upstream'in `models.json` dosyası WSP'de kullanılmaz (HuggingFace
modellerini listeler). WSP kataloğu `wsp/models.json` ve kütüphanedeki
`models.json` dosyalarıdır.
