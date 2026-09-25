# WSP CameraTrap: AddaxAI incelemesi, mimari ve uygulama planı

> Durum: **ONAYLANDI, uygulanıyor.** Güncel kullanım: `USER_GUIDE.md` ve `ADMIN_GUIDE.md`.
> İncelenen upstream: `PetervanLunteren/AddaxAI` `main`, commit `b8b75e2` (23 Eylül 2026), MIT lisanslı.

---

## 1. AddaxAI (son sürüm, v7) incelemesi

### 1.1 Yapı

AddaxAI 7, eski Tkinter uygulamasından (v6) tamamen farklı, yeniden yazılmış bir uygulama:

| Katman | Teknoloji | Boyut |
|---|---|---|
| Masaüstü kabuğu | Electron 34 (`electron/src/main.ts`) | ~1.6k satır TS |
| Arayüz | React + Vite + TanStack + Radix (`frontend/`) | ~64k satır TS |
| Backend | FastAPI + SQLite/Alembic, PyInstaller ile tek exe (`backend/`) | ~55k satır Python |
| ML çalıştırma | Ayrı micromamba/conda ortamları (`backend/app/ml/envs/*`): `addaxai-base` (MegaDetector), `pytorch` (sınıflandırıcılar) | ilk açılışta kurulur |
| Test/CI | pytest, Playwright e2e, GitHub Actions (`tests.yml`, `build-electron.yml`) | kapsamlı |

Çalışma şekli: Electron açılır, PyInstaller ile paketlenmiş backend'i `localhost`'ta başlatır, arayüz onunla konuşur. Analizler, kullanıcı profilindeki conda ortamlarında (`~/AddaxAI/envs/...`) çalışan worker süreçlerinde yapılır.

Özellikler: proje / site / deployment yönetimi, klasör analizi sihirbazı (resim + video), MegaDetector + tür sınıflandırma, insan doğrulaması (verify), taksonomi filtre ağacı, olay (event) gruplama, dashboard, aktivite grafikleri, harita, dışa aktarım (CSV, XLSX, Camtrap DP, Timelapse, EXIF yazma, klasörlere ayırma, işaretli kopyalar), veritabanı yedekleme/geri yükleme.

### 1.2 Model sistemi

- Katalog: `models.json` (7 dedektör, 35 sınıflandırıcı, 3 embedding modeli). Önce `raw.githubusercontent.com`'dan çekiliyor, olmazsa uygulamanın içindeki kopya kullanılıyor.
- Her model, **HuggingFace**'te bir repo: `Addax-Data-Science/<MODEL_ID>`. İçinde ağırlıklar, `inference.py` (sınıflandırıcılar için), `taxonomy.csv` ve SpeciesNet için geofence ve etiket dosyaları var.
- Yerel yerleşim: `~/AddaxAI/models/{det,cls,emb}/<MODEL_ID>/` ve her klasörde bir `manifest.json`. `ManifestManager` bu klasörleri tarıyor ve `manifest.json` olan her klasörü model sayıyor.
- Sınıflandırıcı eklenti sözleşmesi: model klasöründeki `inference.py`, bir `ModelInference` sınıfı tanımlıyor (`load_model`, `get_crop`, `get_classification`, `get_class_names`). Şablonu `backend/templates/inference_template.py`'de. **Kendi modellerimizi tam olarak bu sözleşmeyle ekleyebiliriz.**
- Kurulum sihirbazı (`routers/setup.py`) sırasıyla şunları yapıyor: (1) `addaxai-base` ortamını kuruyor, (2) varsayılan modelleri indiriyor (MD5A ve DINOv2-S, ikisi de HuggingFace'ten).

### 1.3 Dosyalar nereden iniyor? (WSP açısından)

| Host | Ne için | WSP'de durum |
|---|---|---|
| `github.com`, `raw.githubusercontent.com` | installer, katalog | ✅ izinli (senin gözlemin) |
| `micro.mamba.pm`, conda-forge, `pypi.org` / `files.pythonhosted.org`, `download.pytorch.org` | ilk açılışta Python ortamlarının kurulumu | ✅ büyük ihtimalle izinli (aşağıdaki açıklamaya bak) |
| `huggingface.co`, `*.hf.co`, Addax relay (`*.workers.dev`) | **bütün model ağırlıkları**, `inference.py`, `taxonomy.csv` | ❌ **engelli. Sorun tam olarak burada.** |
| `api.github.com`, harita tile sunucuları (OSM/Carto/ArcGIS), `docs.addaxai.com` | güncelleme kontrolü, harita, yardım | kritik değil |

Ortam kurulumu adımı ağırlık indirme adımından **önce** çalışıyor. Sihirbaz ağırlıklarda hata veriyorsa ortam kurulumu (conda-forge, PyPI ve PyTorch indirmeleri) başarıyla bitmiş demektir. AddaxAI'ın kendi dokümanı da bu durumu anlatıyor: bazı kurumların web filtreleri `huggingface.co`'yu "AI content" kategorisinde engelliyor (`DEVELOPERS.md`, "The relay, for networks that block HuggingFace outright").

### 1.4 AddaxAI neden şirket filtresine takılmıyor?

Kaynak koddan çıkan bulgular:

1. **Kullanıcı bazlı kurulum, admin/UAC yok.** `electron/package.json` → `nsis: { oneClick: false }`, `perMachine` tanımlı değil, yani electron-builder varsayılanı kullanılıyor: kurulum `%LOCALAPPDATA%\Programs\AddaxAI` altına, yönetici izni olmadan yapılıyor. `installer.nsh` bunu açıkça yazıyor: *"AddaxAI's electron-builder NSIS install is per-user by default (no UAC)"*. Registry'ye yalnızca `HKCU` altına yazıyor. Servis, sürücü veya `Program Files` yazımı yok. Kurumlarda "sadece IT ticket'la kurulum" kuralı neredeyse her zaman "kullanıcının local admin yetkisi yok" demek: UAC isteyen her installer duruyor, bu installer ise hiç UAC istemiyor.
2. **Authenticode imzalı.** `win.signtoolOptions`: yayıncı "Addax Data Science", sha256, DigiCert timestamp. Bu, SmartScreen uyarısını azaltıyor.
3. **Kullanıcı profilinden imzasız exe çalıştırılabiliyor.** AddaxAI analizleri `~/AddaxAI/envs/.../python.exe` (conda-forge, imzasız) ile çalıştırıyor ve WSP'de çalışıyor. Buradan çıkan sonuç: WSP'de profil klasörlerini kilitleyen katı bir AppLocker/WDAC allowlist'i **yok**.

**Sonuç:** Uygulamamızın kurulum izi AddaxAI ile birebir aynı olmalı: kullanıcı bazlı NSIS installer, admin yok, veriler kullanıcı profilinde. İmza varsa iyi olur ama zorunlu değil. WSP IT'nin iç code-signing sertifikası varsa CI'da onu kullanırız.
**Kalan risk:** IT, AddaxAI'ı yayıncısına (sertifikasına) göre özel olarak allowlist'e almış olabilir. O durumda bizim imzasız build'imiz engellenir. Bu riski Faz 0'daki kanarya testi ilk gün ortaya çıkarır (bkz. §3). Çıkarsa çözümü tek seferlik bir IT ticket ya da WSP sertifikasıyla imzalamak.

### 1.5 Ağırlıklar HuggingFace dışında nereden bulunur?

- **MegaDetector v5a/v5b/v1000**: `github.com/agentmorris/MegaDetector` Releases. Doğruladım: `md_v5a.0.0.pt` (280 MB) GitHub release asset olarak iniyor. GitHub izinli olduğu için **otomatik indirilebilir**.
- **SpeciesNet**: yalnızca Kaggle ve HuggingFace'te var, ikisi de yasak. Şirket ağı dışında **bir kez** indirilip OneDrive kütüphanesine konacak (Apache-2.0, şirket içinde dağıtılabilir). `speciesnet` paketi yerel klasörden yüklemeyi destekliyor (`info.json` + `.pt` + labels + taxonomy + geofence).
- **DINOv2** (benzerlik araması için embedding): isteğe bağlı. Projede `embedding_model_id = "none"` destekleniyor, zorunlu olmaktan çıkaracağız.

### 1.6 Mevcut `Camera_Trap` reposu

Gradio demosu (`app.py`, `model.py`, `app.ipynb`), SpeciesNet ve torchvision adaptörleri, WSP veri hazırlama, doğrulama, eğitim ve benchmark scriptleri ile testler. Git'e commit edilmiş `.pth` ağırlıkları (~56 MB), `__pycache__` ve çıktı klasörleri de var. **Eğitim ve veri hazırlama kısmı değerli**; masaüstü uygulaması kısmı AddaxAI'ın yerini tutmuyor.

---

## 2. Mimari: "WSP CameraTrap" = AddaxAI 7'nin ince bir fork'u

### 2.1 Karar

**AddaxAI 7'yi git geçmişiyle birlikte bu repoya alıp üzerinde küçük ve izole değişiklikler yapacağız.** Uygulamayı sıfırdan yazmayacağız.

| Seçenek | Değerlendirme |
|---|---|
| **A. İnce fork (seçilen)** | Bütün özellikler, testler ve CI hazır geliyor. Değişiklik birkaç dosyada toplanıyor. Upstream düzeltmeleri `git merge upstream/main` ile alınabiliyor. Kalite kanıtlanmış (beta testleri, kurumsal ağ düzeltmeleri). |
| B. Kendi sade uygulamamız (Gradio/PySide) | 120k satırlık özellik setinin küçük bir kısmını yeniden yazmak demek. Daha yavaş, daha düşük kalite, bakım tamamen bize kalıyor. |
| C. AddaxAI'ı olduğu gibi kullanıp modelleri elle koymak | Mümkün (`check_weights_ready` yalnızca dosyanın varlığına bakıyor) ama desteklenmiyor. Upstream güncellemesiyle kırılabilir ve WSP modellerini düzgün dağıtmanın yolu yok. |

Upstream'e yakın kalma kuralı: WSP değişiklikleri az sayıda dosyada olacak, `# WSP:` yorumuyla işaretlenecek ve mümkün olduğunda yeni dosyalara konacak. Böylece upstream birleştirmeleri kolay kalır.

### 2.2 AddaxAI'a göre değişenler (bütün fark bu)

1. **Marka ve izolasyon.** `productName` "WSP CameraTrap", `appId` `com.wsp.cameratrap`, veri klasörü `~/WSP-CameraTrap`, registry anahtarı ve ikon değişiyor. About sayfasına "Based on AddaxAI (MIT)" atfı ekleniyor. Aynı makinede kurulu bir AddaxAI ile **hiç çakışmaz**.
2. **Model kaynağı HuggingFace yerine klasör.** Modeller sırayla şu kaynaklardan aranır:
   1. **Yerel `models/` klasörü** (`~/WSP-CameraTrap/models/{det,cls,emb}/<ID>/`). Klasöre bir model konduğunda uygulama açılışta onu görür ("klasöre ağırlığı koy, çalışsın").
   2. **WSP Model Kütüphanesi** (isteğe bağlı): OneDrive'da senkronize edilen SharePoint klasörü ya da `\\sunucu\paylaşım`. Yolu Ayarlar'dan veya kurulum klasöründeki `wsp-config.json`'dan veriliyor, bilinen OneDrive yollarında otomatik de aranıyor. Kütüphanede bir `models.json` (katalog) ve aynı yerleşimde model klasörleri bulunuyor. Modeller sayfasındaki "Install" düğmesi modeli kütüphaneden yerel klasöre **kopyalar**. İlerleme çubuğu, iptal ve `.tmp` + atomik rename davranışı mevcut HF indiricisiyle aynı. Kütüphanedeki bir dosya değişince "update available" gösterilir.
   3. **URL** (yalnızca tek dosyalık, herkese açık ağırlıklar için, ör. GitHub'daki MegaDetector). Katalog girdisinde `download_url` alanı olarak tutuluyor.
   - Uygulama içinde bu, `HuggingFaceRepoDownloader.download_repo(...)` ile aynı arayüze sahip bir `FolderRepoDownloader` olacak. Değişiklik `model_storage.py` / `catalog_updater.py` / `config.py` ve Ayarlar sayfasındaki bir alanla sınırlı. HF ve relay çağrıları kapatılıyor (hiçbir istek yasak hostlara gitmiyor).
3. **Varsayılan modeller:** MegaDetector v5a (dedektör) ve SpeciesNet v4.0.2a (sınıflandırıcı). SpeciesNet için AddaxAI'ın HF'teki `inference.py`'sini kullanamıyoruz. Onun yerine, mevcut `speciesnet_adapter.py` mantığıyla **kendi `inference.py`'mizi yazacağız** (MD kırpıntısı → `SpeciesNetClassifier`). `taxonomy.csv`'yi AddaxAI'ın `scripts/generate_taxonomy_csv.py` scripti etiket dosyasından üretecek. Geofence için AddaxAI'ın `geofence.py`'si, SpeciesNet klasöründeki geofence dosyasını olduğu gibi okuyor. UK projelerinde ülke GBR seçilince SpeciesNet'in yanlış kıtadan tür önermesi azalıyor. Kurulum sihirbazı yalnızca MD'yi zorunlu tutacak, DINOv2 isteğe bağlı olacak.
4. **WSP modellerini paketleme aracı:** `training/package_model.py`, eğitilmiş bir checkpoint'ten kütüphaneye konmaya hazır bir klasör üretecek: ağırlıklar, `inference.py`, `taxonomy.csv`, sha256 ve `models.json` girdisi (sürüm numarası artırılarak). İki `inference.py` şablonu olacak:
   - `torchvision`: mevcut `mobilenet_v3_small` checkpoint formatı (`wildlife_classifier.py` mantığı).
   - `speciesnet-finetune`: SpeciesNet omurgası üzerinde fine-tune edilmiş model (AddaxAI'ın `addax-sppnet` tipindeki gibi; `pytorch` ortamında `onnx2torch` zaten var).
5. **Python ortamları:** Upstream'deki micromamba kurulumu aynen kalıyor (WSP'de çalıştığına dair kanıt §1.3'te). Kanarya testinde ortam kurulumu başarısız olursa yedek plan hazır: önceden hazırlanmış ortam paketi (`env-pack.zip`) kütüphaneye konur ve uygulama indirme yerine onu açar. **İhtiyaç çıkmazsa bunu yapmayacağız.**
6. **Build ve dağıtım:** AddaxAI'ın `build-electron.yml` workflow'u uyarlanacak. İlk hedef yalnızca Windows, macOS isteğe bağlı. Çıktı `WSP-CameraTrap-Setup-<ver>.exe` (kullanıcı bazlı NSIS). İmza isteğe bağlı: `WSP_PFX` secret'ı varsa imzalanır, yoksa imzasız. Artifact'ler GitHub Release'e konuyor, sen oradan OneDrive'a kopyalıyorsun.
7. **Gereksiz dış bağlantılar kapatılıyor:** güncelleme kontrolü (AddaxAI'ın GitHub releases'ına bakıyor) ve katalog senkronizasyonu kapanıyor. Harita tile'ları kalıyor (engelliyse yalnızca harita arka planı boş görünür).

### 2.3 Klasör yerleşimi

```
OneDrive / SharePoint: "WSP CameraTrap"            (herkes okur, yalnızca sen yazarsın)
├── README.txt                                     (kurulum talimatları)
├── installer/WSP-CameraTrap-Setup-1.0.0.exe
└── models/
    ├── models.json                                (katalog; package_model.py günceller)
    ├── det/MD5A-0-0/md_v5a.0.0.pt
    ├── cls/SPECIESNET-v4-0-2-A/                   (.pt, labels, taxonomy.csv, geofence, info.json, inference.py)
    └── cls/WSP-UK-v1/                             (model.pt, inference.py, taxonomy.csv)

Kullanıcı bilgisayarı
├── %LOCALAPPDATA%\Programs\WSP CameraTrap\        (uygulama; admin gerekmez)
└── %USERPROFILE%\WSP-CameraTrap\                  (models/, envs/, logs/, backups/, wsp.db)
```

Repo yerleşimi (fork sonrası):

```
backend/  electron/  frontend/  docs/  infra/      ← AddaxAI (upstream) + küçük WSP değişiklikleri
wsp/                                               ← WSP'ye özel: varsayılan katalog, SpeciesNet inference.py, config örneği
training/                                          ← mevcut veri hazırlama, eğitim, benchmark scriptleri ve testleri + package_model.py
docs/WSP_*.md                                      ← bu plan, kullanıcı kılavuzu, model yayınlama kılavuzu
```

Mevcut Gradio demosu (`app.py`, `model.py`, `app.ipynb`, `new_model_test.py`, `repo_bundle.md`), `batch_output/`, demo çıktıları ve `__pycache__` silinecek (git geçmişinde duruyorlar). `.pth` dosyaları git'ten çıkarılıp `.gitignore`'a eklenecek. Senin bilgisayarındaki kopyalar yerinde kalır.

### 2.4 Kullanıcı akışları

**Ekolog (ilk kurulum):** OneDrive'da "WSP CameraTrap" klasörünü senkronize et ("Add shortcut to My files") → `installer/…Setup.exe`'yi çalıştır (admin gerekmez) → ilk açılışta sihirbaz Python ortamını kurar (bir kerelik, internet gerekir, 10–20 dk) ve kütüphaneyi otomatik bulur → MD ve SpeciesNet'i kopyalar → kullanıma hazır.

**Sen (yeni model yayınlama):** Modeli eğit → `python training/package_model.py --checkpoint … --id WSP-UK-v2 --library "<OneDrive yolu>"` → OneDrive senkronize eder → ekolog uygulamayı açınca "WSP-UK-v2 available" görür → Install.

**İnternetsiz / kütüphanesiz senaryo:** Model klasörünü elle `~/WSP-CameraTrap/models/cls/` altına kopyala. Uygulama onu da görür.

---

## 3. Uygulama planı

Her faz ayrı commit'lerle bu branch'te (`claude/wsp-speciesnet-app-0gendb`) ilerleyecek. Her fazın sonunda backend testleri (`pytest`) ve lint çalıştırılacak. Upstream testleri kırılmayacak, yeni kod için yeni testler yazılacak.

### Faz 0: Temel ve risk giderme (önce bu)
0.1. Upstream'i al: `git remote add upstream …AddaxAI` → `git merge --allow-unrelated-histories upstream/main` (git geçmişi yaklaşık 26 MB). MIT lisansı ve atıf korunacak.
0.2. Mevcut dosyaları yeniden düzenle: eğitim ve veri scriptleri ile testleri `training/`'e taşı, Gradio demosunu ve çıktıları sil, `.pth` dosyalarını git'ten çıkar.
0.3. Marka ve izolasyon (§2.2-1): package.json, config.py (`~/WSP-CameraTrap`), installer.nsh (HKCU anahtarı; Timelapse shim kaldırılır), pencere başlıkları, About sayfası.
0.4. CI: Windows installer build workflow'u (imzasız, isteğe bağlı imzalı). Tag'lendiğinde GitHub Release'e yükler.
0.5. **Kanarya testi (senin yapacağın adım):** Bir WSP laptopunda installer'ı çalıştır: admin istemeden kuruluyor mu, açılıyor mu, ortam kurulumu bitiyor mu? → **Karar noktası:** engel çıkarsa imza veya IT ticket yoluna geçeriz, ortam kurulumu takılırsa env-pack yedeğine geçeriz.

### Faz 1: Klasör tabanlı model kaynağı (çekirdek)
1.1. `Settings`: `model_library_dir` (Ayarlar'dan veya `wsp-config.json`'dan) ve bilinen OneDrive yollarında otomatik arama. HF ve relay istekleri kapatılıyor.
1.2. `FolderRepoDownloader`: kopyalama, ilerleme, iptal, `.tmp` + atomik rename, eksik dosyaları tamamlama.
1.3. `download_url` kaynağı (tek dosya, GitHub).
1.4. Katalog: kütüphanedeki `models.json` → uygulamayla gelen WSP `models.json`. Güncelleme kontrolü dosya boyutu ve sha256'ya göre yapılıyor.
1.5. Arayüz: Ayarlar'da "Model library folder" alanı, kurulum sihirbazındaki hata mesajları HF yerine kütüphaneyi anlatıyor.
1.6. Testler: downloader (kopyalama/iptal/eksik dosya), katalog önceliği, "klasöre konan model görünür" senaryosu.

### Faz 2: Varsayılan modeller
2.1. MegaDetector v5a: katalogda kütüphane ve GitHub URL kaynakları.
2.2. SpeciesNet v4.0.2a: `wsp/models/SPECIESNET/inference.py` (speciesnet paketiyle MD kırpıntısını sınıflandırıyor), `taxonomy.csv` üretimi, `speciesnet` paketinin `pytorch` env YAML'ına eklenmesi, geofence doğrulaması.
2.3. Kurulum sihirbazı: yalnızca MD zorunlu, DINOv2 isteğe bağlı.
2.4. Uçtan uca doğrulama: `demo_data` resimleriyle MD ve SpeciesNet analizi (burada CPU'da çalışacak, ağırlıklar olmadan yalnızca birim testleri; tam test senin makinende).

### Faz 3: WSP model hattı
3.1. `training/package_model.py` ve iki `inference.py` şablonu (torchvision, speciesnet-finetune).
3.2. Mevcut `train_species_classifier.py` çıktısının paketleyiciyle uyumlu olması (sınıf adları, normalizasyon, eşikler checkpoint'te).
3.3. Testler: paketlenen modelin `ModelInference` sözleşmesine uyması (sahte küçük modelle).

### Faz 4: Dokümantasyon ve 1.0 sürümü
4.1. `docs/WSP_USER_GUIDE.md` (ekologlar için: kurulum, ilk analiz, doğrulama, dışa aktarım) ve kütüphanedeki `README.txt`.
4.2. `docs/WSP_ADMIN_GUIDE.md` (senin için: OneDrive kütüphanesini kurma, SpeciesNet'i bir kez indirme, model yayınlama, sürüm çıkarma, upstream'den güncelleme alma).
4.3. `v1.0.0` tag'i → CI installer'ı üretir.

### Benim yapamayacağım, sana kalan işler
- WSP laptopunda kanarya testi (Faz 0.5) ve son kullanıcı testi.
- SpeciesNet ağırlıklarını şirket ağı dışında bir kez indirmek (Kaggle `google/speciesnet/pyTorch/v4.0.2a`) ve MD ağırlığıyla birlikte OneDrive klasörüne koymak.
- OneDrive/SharePoint klasörünü oluşturup ekiple paylaşmak.
- (İsteğe bağlı) WSP IT'den code-signing sertifikası.

### Onayda netleştirmek istediklerim (cevap gelmezse varsayılanla ilerlerim)
1. Uygulama adı: varsayılan **"WSP CameraTrap"**.
2. Platform: varsayılan **yalnızca Windows** (macOS build workflow'da kalır ama ilk sürümde test edilmez).
3. SpeciesNet sürümü: varsayılan **v4.0.2a** (AddaxAI'daki `always_crop` sürümü, MD kırpıntılarıyla uyumlu).
