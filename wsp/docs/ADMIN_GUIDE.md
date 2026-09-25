# WSP CameraTrap: yönetici kılavuzu

Bu kılavuz uygulamayı yayınlayan ve modelleri dağıtan kişi içindir (sen).
Ekolog arkadaşların için olan kılavuz `USER_GUIDE.md`.

## Genel yapı

```
GitHub: cameaci/Camera_Trap                OneDrive (senin hesabın)
  ├─ kaynak kod                               └─ WSP-CameraTrap-models.zip
  ├─ Release "v1.0.0" ...                          (paylaşım linki → wsp/config.json)
  │    └─ WSP-CameraTrap-Setup-1.0.0.exe             ├─ models.json
  └─ Release "runtime"  (uygulama ilk açılışta indirir) ├─ det/MD5A-0-0/…
       ├─ env-wsp-base-win-64-<hash>.json/.zip.001     ├─ cls/SPECIESNET-v4-0-2-A/…
       └─ md_v5a.0.0.pt                                └─ cls/WSP-UK-v1/…
```

Uygulama internetten yalnızca iki yere gider:

1. **Bu repo'nun GitHub Releases'ı**: installer, analiz ortamı
   (Python + CPU torch, `runtime` release'inde, CI üretir) ve yedek
   MegaDetector.
2. **Senin OneDrive linkin**: model kütüphanesi (`.zip`).

HuggingFace, Kaggle, conda-forge veya PyPI'ye bağlanmaz (analiz ortamı
paketi bulunamazsa son çare olarak micromamba ile kurmayı dener).

Model kütüphanesi şu sırayla aranır:

1. `WSP_MODEL_LIBRARY_DIR` ortam değişkeni (test için)
2. Kullanıcının *File › WSP model library…* ekranında seçtiği klasör
3. **Link**: kullanıcının girdiği link, yoksa installer'a gömülü
   `wsp/config.json › model_library_url`
4. OneDrive'da senkronize edilmiş bir `WSP CameraTrap/models` klasörü

## 1. Model kütüphanesini hazırlamak (bir kez)

Kendi bilgisayarında, repo klonunda:

```bash
LIB=~/wsp-library/models
python wsp/tools/wsp_library.py init "$LIB"
```

**MegaDetector v5a**: `md_v5a.0.0.pt` dosyasını bu repo'nun `runtime`
release'inden indir
(https://github.com/cameaci/Camera_Trap/releases/download/runtime/md_v5a.0.0.pt)
ve ekle:

```bash
python wsp/tools/wsp_library.py add-md "$LIB" md_v5a.0.0.pt
```

(Kütüphanede MegaDetector olmasa da uygulama onu `runtime` release'inden
indirir. Yine de kütüphaneye koymak en güvenlisi.)

**SpeciesNet v4.0.2a**: Kaggle ve HuggingFace şirket ağında engelli olduğu
için bu dosyaları **bir kez şirket ağı dışında** indir: Kaggle'da
`google/speciesnet`, *PyTorch*, sürüm **v4.0.2a**. Klasörde
`always_crop_99710272_22x8_v12_epoch_00148.pt`, `…labels.txt`,
`geofence_release…json`, `info.json` ve `taxonomy_release.txt` bulunur.
Sonra:

```bash
python wsp/tools/wsp_library.py add-speciesnet "$LIB" path/to/speciesnet-v4.0.2a
```

Bu komut dosyaları kopyalar, `inference.py` dosyasını ekler, `taxonomy.csv`
dosyasını üretir ve katalog girdisini yazar. SpeciesNet Apache-2.0,
MegaDetector MIT lisanslıdır; ikisi de şirket içinde dağıtılabilir.

## 2. Kütüphaneyi OneDrive'a koymak ve linki uygulamaya vermek

1. Kütüphaneyi tek bir zip yap:

   ```bash
   python wsp/tools/wsp_library.py bundle "$LIB" WSP-CameraTrap-models.zip
   ```

2. Zip'i OneDrive'a yükle. **Share › Anyone with the link can view** ile
   link oluştur. (Şirket politikası "Anyone" linkine izin vermiyorsa
   "People in WSP" linki tarayıcıda çalışır ama uygulama oturum açamaz; o
   durumda arkadaşların klasörü senkronize edip *Use a folder…* seçmeli.)
3. Linki `wsp/config.json` dosyasına yaz:

   ```json
   { "model_library_url": "https://wsponline-my.sharepoint.com/:u:/g/personal/…" }
   ```

   Uygulama linke `download=1` ekler ve dosyayı doğrudan indirir.
4. Değişikliği commit'le ve yeni bir installer çıkar (bölüm 4). Bu installer'ı
   kuran herkes modelleri ilk açılışta otomatik indirir.

Link'i installer çıkarmadan denemek istersen: uygulamada *File › WSP model
library…* ekranına yapıştır ve **Connect**'e bas.

## 3. Yeni model yayınlamak (aşamalı dağıtım)

1. Modeli eğit (`training/train_species_classifier.py`). Çıktı
   `state_dict`, `class_names`, `arch`, `input_size` ve `normalization`
   içeren bir `.pth` dosyasıdır.
2. Kütüphaneye ekle. **Her yeni sürüm için yeni bir id kullan** (`WSP-UK-v1`,
   `WSP-UK-v2`, …). Eski sürümü kullananlar bozulmaz, yeni sürüm listede
   ayrı bir model olarak görünür.

   ```bash
   python wsp/tools/wsp_library.py add-model "$LIB" \
       --checkpoint training/wsp_uk_v2.pth \
       --id WSP-UK-v2 --name "WSP UK mammals v2" \
       --taxonomy uk_taxonomy.csv        # isteğe bağlı: model_class,class,order,family,genus,species
   ```

3. `bundle` ile zip'i yeniden üret ve OneDrive'daki dosyanın **üzerine
   yükle** (aynı dosya, aynı link). Yeni installer gerekmez: uygulamalar bir
   sonraki açılışta (ya da *Check for new models* ile) değişikliği görür,
   zip'i indirir ve yeni modeli **Install** butonuyla sunar.

`inference.py`'yi veya `taxonomy.csv`'yi aynı id altında düzeltirsen
uygulama bunu "update available" olarak gösterir. Ağırlık dosyasını aynı id
altında değiştirme; yeni bir id yayınla.

Paketleme aracı eğitim scriptinin `mobilenet_v3_small` mimarisini
destekliyor. SpeciesNet üzerine fine-tune edeceğin modeller için
`wsp/models/SPECIESNET-v4-0-2-A/inference.py` başlangıç noktasıdır: aynı ön
işleme, farklı ağırlık ve etiket dosyası.

## 4. Uygulamanın yeni sürümünü çıkarmak

1. Değişiklikler `main` dalında olsun ve CI yeşil olsun.
2. Bir tag it: `git tag v1.0.0 && git push origin v1.0.0`
   (ya da GitHub'da **Releases › Draft a new release**).
3. `Build Windows installer` workflow'u installer'ı üretir ve release'e
   `WSP-CameraTrap-Setup-1.0.0.exe` olarak ekler (yaklaşık 20 dakika). Tag'de
   `-` varsa (`v1.0.0-beta.1`) pre-release olarak çıkar.
4. Arkadaşlarına release sayfasının linkini gönder.

**Analiz ortamı.** `backend/app/ml/envs/wsp-base/windows/environment.yml`
değişirse `Build runtime downloads` workflow'u yeni ortam paketini üretip
`runtime` release'ine yükler (yaklaşık 30 dakika). Paket adı YAML'ın
hash'ini taşır; uygulama yalnızca kendi YAML'ına uyan paketi kurar.
`runtime` release'ini silme.

**İmza.** WSP IT'den bir code-signing sertifikası (`.pfx`) alırsan, repo
ayarlarında `WSP_PFX_BASE64` ve `WSP_PFX_PASSWORD` secret'larını tanımla.
Bundan sonra installer imzalı çıkar ve SmartScreen uyarısı kaybolur.
