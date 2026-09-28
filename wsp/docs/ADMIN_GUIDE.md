# WSP CameraTrap: yönetici kılavuzu

Bu kılavuz uygulamayı yayınlayan ve modelleri dağıtan kişi içindir (sen).
Ekolog arkadaşların için olan kılavuz `USER_GUIDE.md`.

## Genel yapı

```
GitHub: cameaci/Camera_Trap                     OneDrive (senin hesabın)
  ├─ kaynak kod                                    └─ WSP-CameraTrap-models.zip
  ├─ Release "v0.1.x"                                  (arkadaşların indirir ve
  │    └─ WSP-CameraTrap-Setup-0.1.x.exe                uygulamada kurar)
  ├─ Release "models"  → başlangıç zip'i (MegaDetector, SpeciesNet, DINOv2)
  └─ Release "runtime" → analiz ortamı + MegaDetector (ilk açılışta iner)
```

Uygulama internetten yalnızca bu repo'nun GitHub Releases'ına gider
(analiz ortamı ve yedek MegaDetector). HuggingFace, Kaggle, conda-forge
veya PyPI'ye bağlanmaz.

Modeller **zip dosyasıyla** dağıtılır: kullanıcı zip'i OneDrive'dan (ya da
herhangi bir yerden) indirir ve uygulamada **File › WSP model library… ›
Install models from a .zip…** ile seçer. Uygulama zip'i
`%USERPROFILE%\WSP-CameraTrap\models` klasörüne açar. Zip'in içindeki
`models.json` o klasörde tutulur ve katalogla birleştirilir; bu sayede zip'e
eklediğin yeni bir model, yeni uygulama sürümü gerekmeden görünür.

İsteğe bağlı: kütüphaneyi zip yerine senkronize bir klasör (OneDrive ya da
ağ klasörü) olarak da kullanabilirsin: aynı ekranda **Use a folder…**.

## 1. Model kütüphanesi

Başlangıç kütüphanesi hazır: GitHub'daki `models` release'inde
`WSP-CameraTrap-models.zip` (MegaDetector v5a, SpeciesNet 4.0.2a, DINOv2
ViT-S/14). `Build model library` workflow'u onu bu repo'daki araçla üretir
ve modelleri uygulamanın kendi koduyla yükleyip dener. SpeciesNet ve DINOv2
Apache-2.0, MegaDetector MIT lisanslıdır; şirket içinde dağıtılabilir.

Kendi modellerini eklemek için bu zip'i kendi bilgisayarında bir klasöre aç.
İçindeki `models` klasörü kütüphanenin kendisidir:

```bash
LIB=~/wsp-library/models      # zip'ten çıkan models klasörü
```

Sıfırdan başlamak istersen: `python wsp/tools/wsp_library.py init "$LIB"`,
sonra `add-md`, `add-speciesnet` ve `add-dinov2` komutları (bkz.
`python wsp/tools/wsp_library.py --help`).

## 2. Zip'i dağıtmak

1. Kütüphaneyi tek bir zip yap:

   ```bash
   python wsp/tools/wsp_library.py bundle "$LIB" WSP-CameraTrap-models.zip
   ```

2. Zip'i OneDrive'daki bir klasöre koy ve arkadaşlarınla paylaş (şirket içi
   paylaşım yeterli; linke uygulama değil, kişi tıklar).
3. Arkadaşların zip'i indirip uygulamada **File › WSP model library… ›
   Install models from a .zip…** ile kurar.

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

3. `bundle` ile zip'i yeniden üret ve OneDrive'daki dosyanın üzerine yükle.
   Arkadaşların yeni zip'i aynı menüden kurar; eski modelleri silinmez, yeni
   model listede görünür. Yeni installer gerekmez.

`inference.py`'yi veya `taxonomy.csv`'yi aynı id altında düzeltirsen yeni
zip onları günceller. Ağırlık dosyasını aynı id altında değiştirme; yeni bir
id yayınla.

Paketleme aracı eğitim scriptinin `mobilenet_v3_small` mimarisini
destekliyor. SpeciesNet üzerine fine-tune edeceğin modeller için
`wsp/models/SPECIESNET-v4-0-2-A/inference.py` başlangıç noktasıdır: aynı ön
işleme, farklı ağırlık ve etiket dosyası.

## 4. Uygulamanın yeni sürümünü çıkarmak

1. Değişiklikler `main` dalında olsun ve CI yeşil olsun.
2. Repo kökündeki `VERSION` dosyasını yeni sürüme değiştir (`1.0.0`) ve
   commit'le. Alternatif: `git tag v1.0.0 && git push origin v1.0.0`, ya da
   GitHub'da **Releases › Draft a new release**.
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
