# Remedia — Reseptör Odaklı İlaç Keşif Pipeline'ı

GPU üzerinde çalışan, reseptör hedefli açık kaynak ilaç keşif prototipi:

**Molekül üretimi → AlphaFold DB → pocket detection → GNINA GPU docking → drug-likeness filtresi → sıralama**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb)
[![Local GPU Notebook](https://img.shields.io/badge/Local-NVIDIA_GPU-2E7D32)](notebooks/remedia_local.ipynb)
[![Open Modal Notebooks](https://img.shields.io/badge/Modal-Open_Notebook-7F5AF0)](https://modal.com/notebooks)
[![Run on RunPod](https://img.shields.io/badge/RunPod-Deploy_GPU-673DE6?logo=runpod&logoColor=white)](https://console.runpod.io/deploy)

## Yerel bilimsel çalışma istasyonu

Yerel akış şu sistemler içindir:

- Linux x86_64 + NVIDIA GPU
- Windows 11 + WSL2 + NVIDIA GPU
- NVIDIA GPU düğümü bulunan üniversite/HPC Linux sistemi
- En az 8 GB RAM ve yaklaşık 20 GB boş disk önerilir

macOS ve NVIDIA GPU bulunmayan bilgisayarlarda mevcut CUDA/GNINA akışı yerel
çalışmaz; Colab, Modal veya RunPod kullanılmalıdır.

### Otomatik kurulum

```bash
git clone https://github.com/mehmetg06/Remedia.git
cd Remedia
bash scripts/setup_local.sh
```

Script kendi izole micromamba ortamını `.remedia-tools/env` altında oluşturur,
`environment.yml` bağımlılıklarını kurar, fpocket ve GNINA'yı doğrular, Jupyter
kernel'ini kaydeder ve GPU gerektirmeyen testleri çalıştırır.

Kurulum sonunda ekrana yazılan Jupyter komutunu çalıştır. Notebook:

```text
notebooks/remedia_local.ipynb
```

Yerel sonuçlar ve cache:

```text
local_workspace/Remedia_results/
local_workspace/remedia_cache/pocket_cache.json
```

### Conda ortamını elle kurma

Python bağımlılıklarını ayrı yönetmek isteyen kullanıcılar:

```bash
conda env create -f environment.yml
conda activate remedia-local
```

Bu yöntem Python/fpocket ortamını kurar. GNINA binary'si ve NVIDIA sürücüsü yine
gereklidir; tam otomatik yol için `scripts/setup_local.sh` önerilir.

### Docker ile yerel GPU

Docker hostunda NVIDIA Container Toolkit ve çalışan bir NVIDIA sürücüsü olmalıdır.

```bash
docker build -f Dockerfile.local -t remedia-local .
docker run --rm --gpus all \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD/local_workspace:/workspace/Remedia/local_workspace" \
  remedia-local
```

Tarayıcıda `http://127.0.0.1:8888` açılır. Jupyter token:

```text
remedia
```

Docker imajı CUDA runtime, GNINA, fpocket, RDKit ve JupyterLab'i içerir. Sonuçlar
hosttaki `local_workspace/` klasöründe kalır.

## Modal — form kullanan GPU akışı

1. [Modal Notebooks](https://modal.com/notebooks) sayfasında yeni notebook oluştur.
2. [`notebooks/remedia_modal.ipynb`](notebooks/remedia_modal.ipynb) dosyasını indirip yükle.
3. Compute panelinden **L4**, **4 CPU** ve **8 GiB RAM** seç.
4. Kalıcı sonuç için `remedia-data` adlı Volume oluşturup `/mnt/remedia-data` yoluna bağla.
5. Notebook'taki tek kod hücresini çalıştır.
6. Açılan formdan reseptör, UniProt ID, molekül sayısı ve doğruluk profilini seç.
7. **Remedia'yı Başlat** düğmesine bas.

Notebook eksik Python paketlerini, fpocket'i, GNINA'yı ve Remedia kodunu kendi
kendine kurar. Volume bağlıysa araç cache'i, pocket cache ve sonuçlar sonraki
oturumlarda korunur.

Özel Modal imajı ile daha hızlı tekrar açılış:

```bash
git clone https://github.com/mehmetg06/Remedia.git
cd Remedia
python -m pip install modal
python -m modal setup
modal run modal/remedia_modal.py --timeout-minutes 60
```

Ayrıntılar: [`modal/README.md`](modal/README.md)

## RunPod — hızlı GPU akışı

RunPod sürümü Colab kurulum hücrelerini kullanmaz. Özel imaj GNINA, fpocket,
RDKit ve JupyterLab'i önceden içerir; Pod açıldığında doğrudan
`notebooks/remedia_runpod.ipynb` açılır.

Önerilen ayarlar:

- Image: `ghcr.io/mehmetg06/remedia-runpod:latest`
- GPU: **L40S**; daha ekonomik kullanım için RTX 4090/5090 veya A5000
- HTTP port: `8888`
- Container disk: `20 GB`
- Volume disk: `20 GB`
- Volume mount: `/workspace`
- Jupyter token: `remedia`

Özel imaj kullanmadan resmi RunPod PyTorch template'iyle Web Terminal'de:

```bash
curl -fsSL https://raw.githubusercontent.com/mehmetg06/Remedia/main/runpod/bootstrap.sh | bash
```

Ayrıntılar: [`runpod/README.md`](runpod/README.md)

## Hızlandırılmış varsayılan akış

Notebook'lar günlük geliştirme için hafif ayarlarla açılır:

- `TOP_FRACTION = 0.10`
- `ACCURACY_PROFILE = "balanced"`
- `INSTALL_REINVENT4 = False`
- `RUN_BENCHMARK = False`

GNINA ligand başına ayrı süreç açmaz. Bütün ligandlar FAST aşamasında tek bir
çoklu SDF dosyasına yazılır ve **tek GNINA sürecinde** docklanır. Seçilen en iyi
adaylar ACCURATE aşamasında ikinci bir batch süreçte çalışır. Normal iki aşamalı
tarama toplamda en fazla iki GNINA süreci kullanır.

Ligandların 3D SDF dosyaları yalnızca bir kez hazırlanır. Accurate aşaması FAST
aşamasında hazırlanan aynı konformasyonları yeniden kullanır.

## Accuracy profilleri

| Profil | FAST | ACCURATE | Kullanım |
|---|---|---|---|
| `balanced` | exhaustiveness 4, 1 pose, fast CNN | exhaustiveness 8, 3 pose, varsayılan CNN ensemble | Günlük çalışmalar |
| `final` | exhaustiveness 4, 1 pose, fast CNN | exhaustiveness 16, 9 pose, varsayılan CNN ensemble | Son doğrulama |

`final` profili belirgin biçimde daha yavaştır; yalnızca nihai aday setinde
kullanılması önerilir.

## Pocket cache

```text
Colab:  /content/drive/MyDrive/remedia_setup/pocket_cache.json
Local:  local_workspace/remedia_cache/pocket_cache.json
Modal:  /mnt/remedia-data/remedia_cache/pocket_cache.json
RunPod: /workspace/remedia_cache/pocket_cache.json
```

Yeni hedef ilk kez kullanıldığında fpocket çalışır. Aynı hedef sonraki
çalıştırmalarda cache'den okunur.

## Notebook kullanımı

### Yerel

1. `bash scripts/setup_local.sh` çalıştır.
2. `notebooks/remedia_local.ipynb` dosyasını `Remedia Local (GPU)` kernel'iyle aç.
3. İlk hücrede UniProt ID ve deney parametrelerini ayarla.
4. **Run All Cells** seç.
5. `local_workspace/Remedia_results/` altındaki `final_ranking.csv` dosyasını incele.

### Colab

1. **Runtime → Change runtime type → T4 GPU** seç.
2. Notebook'u `Run all` ile çalıştır.
3. Yeni hedefte Miniconda kernel'i yeniden başlatırsa `Run all`ı tekrar çalıştır.

### Modal

1. Modal Notebook'ta L4 GPU seç.
2. `remedia_modal.ipynb` dosyasını yükle.
3. Mümkünse `/mnt/remedia-data` yoluna Volume bağla.
4. Form hücresini çalıştır ve yeşil başlatma düğmesine bas.

### RunPod

1. NVIDIA GPU'lu Pod başlat ve JupyterLab'i aç.
2. `Remedia/notebooks/remedia_runpod.ipynb` notebook'unda **Run All Cells** seç.
3. İşin bitince Pod'u durdur veya sonlandır.

## Repository yapısı

```text
notebooks/remedia_pipeline.ipynb  Colab akışı
notebooks/remedia_local.ipynb     Yerel Linux/WSL2 NVIDIA GPU akışı
notebooks/remedia_modal.ipynb     Modal form akışı
notebooks/remedia_runpod.ipynb    RunPod Jupyter akışı
scripts/setup_local.sh            Yerel ortam, GNINA ve kernel kurulumu
environment.yml                   Conda/micromamba bağımlılıkları
Dockerfile.local                  Yerel NVIDIA Docker/Jupyter imajı
modal/remedia_modal.py            Modal L4/L40S Jupyter başlatıcısı
modal/requirements.txt            Modal özel imaj bağımlılıkları
runpod/Dockerfile                 GNINA/fpocket hazır RunPod imajı
runpod/bootstrap.sh               RunPod tek komut kurulum
src/gnina_engine.py               Batch GNINA motoru ve accuracy profilleri
src/molecule_generator.py         Fusion, genetic, BRICS ve random üretim
src/generative_model.py           Opsiyonel REINVENT4 sampling
src/fetch_structure.py            AlphaFold/PDB yapı indirme
src/pocket_detection.py           fpocket entegrasyonu
src/admet_filter.py               Lipinski/Veber drug-likeness filtresi
src/rank_report.py                Docking ve filtre sonuçlarını sıralama
tests/test_gnina_engine.py        GNINA motoru testleri
tests/test_modal_assets.py        Modal dosyaları ve notebook syntax testleri
tests/test_local_assets.py        Yerel kurulum dosyaları ve notebook testleri
legacy/                           Eski akışlar
```

## Testler

GNINA binary'si veya GPU gerektirmeden:

```bash
python -m unittest discover -s tests -v
```

Yerel kurulum scriptinin shell syntax kontrolü:

```bash
bash -n scripts/setup_local.sh
```

## Bilimsel kullanım notu

Remedia bir araştırma ve ön eleme prototipidir. Docking skoru tek başına bağlanma
veya etkinlik kanıtı değildir. Son adaylar deneysel yapı, reseptör hazırlama,
pozitif/negatif kontroller, alternatif scoring yöntemleri ve laboratuvar
doğrulamasıyla değerlendirilmelidir.

Tekrarlanabilir çalışmalar için kullanılan commit SHA'sı, UniProt ID, pocket
koordinatları, GNINA profili, rastgele tohumlar ve çıktı dosyaları deney kaydına
eklenmelidir.

## Lisans

AGPL-3.0-or-later. Ayrıntılar için [LICENSE](LICENSE).
