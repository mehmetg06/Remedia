# Remedia — Reseptör Odaklı İlaç Keşif Pipeline'ı

GPU üzerinde çalışan, reseptör hedefli açık kaynak ilaç keşif prototipi:

**Molekül üretimi → AlphaFold DB → pocket detection → GNINA GPU docking → drug-likeness filtresi → sıralama**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb)
[![Open Modal Notebooks](https://img.shields.io/badge/Modal-Open_Notebook-7F5AF0)](https://modal.com/notebooks)
[![Run on RunPod](https://img.shields.io/badge/RunPod-Deploy_GPU-673DE6?logo=runpod&logoColor=white)](https://console.runpod.io/deploy)

## Modal — ücretsiz kredi ve harcama korumalı akış

Modal sürümü iki şekilde kullanılabilir.

### En kolay: Modal Notebooks

1. [Modal Notebooks](https://modal.com/notebooks) sayfasında yeni notebook oluştur.
2. [`notebooks/remedia_modal.ipynb`](notebooks/remedia_modal.ipynb) dosyasını indirip yükle.
3. Compute panelinden GPU olarak **L4** seç.
4. Kalıcı sonuç için `remedia-data` adlı Volume oluşturup `/mnt/remedia-data` yoluna bağla.
5. **Run all** seç.

Notebook eksik Python paketlerini, fpocket'i, GNINA'yı ve Remedia kodunu kendi
kendine kurar. Volume bağlıysa araç cache'i, pocket cache ve sonuçlar sonraki
oturumlarda korunur. Volume bağlı değilse notebook yine çalışır ancak kernel
kapanınca dosyalar silinir.

### Daha hızlı tekrar açılış: özel Modal imajı

```bash
git clone https://github.com/mehmetg06/Remedia.git
cd Remedia
python -m pip install modal
python -m modal setup
modal run modal/remedia_modal.py --timeout-minutes 60
```

Bu komut L4 GPU'lu, token korumalı JupyterLab bağlantısı açar. Oturum varsayılan
olarak 60 dakika sonra otomatik kapanır; izin verilen üst sınır 240 dakikadır.

L40S kullanmak için:

```bash
REMEDIA_MODAL_GPU=L40S modal run modal/remedia_modal.py --timeout-minutes 60
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

Özel imaj kullanmadan resmi RunPod PyTorch template'iyle başlatmak için Web
Terminal'de:

```bash
curl -fsSL https://raw.githubusercontent.com/mehmetg06/Remedia/main/runpod/bootstrap.sh | bash
```

Ayrıntılar: [`runpod/README.md`](runpod/README.md)

## Hızlandırılmış varsayılan akış

Notebook günlük geliştirme için hafif ayarlarla açılır:

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
Modal:  /mnt/remedia-data/remedia_cache/pocket_cache.json
RunPod: /workspace/remedia_cache/pocket_cache.json
```

Yeni hedef ilk kez kullanıldığında fpocket çalışır. Aynı hedef sonraki
çalıştırmalarda cache'den okunur.

## Notebook kullanımı

### Colab

1. **Runtime → Change runtime type → T4 GPU** seç.
2. Notebook'u `Run all` ile çalıştır.
3. Yeni hedefte Miniconda kernel'i yeniden başlatırsa `Run all`ı tekrar çalıştır.

### Modal

1. Modal Notebook'ta L4 GPU seç.
2. `remedia_modal.ipynb` dosyasını yükle.
3. Mümkünse `/mnt/remedia-data` yoluna Volume bağla.
4. **Run all** seç ve idle shutdown değerini düşük bırak.

### RunPod

1. NVIDIA GPU'lu Pod başlat ve JupyterLab'i aç.
2. `Remedia/notebooks/remedia_runpod.ipynb` notebook'unda **Run All Cells** seç.
3. İşin bitince Pod'u durdur veya sonlandır.

## Repository yapısı

```text
notebooks/remedia_pipeline.ipynb  Colab akışı
notebooks/remedia_modal.ipynb     Modal Notebook akışı
notebooks/remedia_runpod.ipynb    RunPod Jupyter akışı
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
legacy/                           Eski akışlar
```

## Testler

GNINA binary'si veya GPU gerektirmeden:

```bash
python -m unittest discover -s tests -v
```

## Bilimsel kullanım notu

Remedia bir araştırma ve ön eleme prototipidir. Docking skoru tek başına bağlanma
veya etkinlik kanıtı değildir. Son adaylar deneysel yapı, reseptör hazırlama,
uygun kontroller ve laboratuvar doğrulamasıyla değerlendirilmelidir.

## Lisans

AGPL-3.0-or-later. Ayrıntılar için [LICENSE](LICENSE).
