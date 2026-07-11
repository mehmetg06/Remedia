# Remedia — Reseptör Odaklı İlaç Keşif Pipeline'ı

GPU üzerinde çalışan, reseptör hedefli açık kaynak ilaç keşif prototipi:

**Molekül üretimi → AlphaFold DB → pocket detection → GNINA GPU docking → drug-likeness filtresi → sıralama**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb)
[![Run on RunPod](https://img.shields.io/badge/RunPod-Deploy_GPU-673DE6?logo=runpod&logoColor=white)](https://console.runpod.io/deploy)

## RunPod — daha hızlı GPU akışı

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

Sonuçlar ve pocket cache kalıcı volume'da tutulur:

```text
/workspace/Remedia_results/
/workspace/remedia_cache/
```

Özel imaj kullanmadan resmi RunPod PyTorch template'iyle başlatmak için Web
Terminal'de tek komut:

```bash
curl -fsSL https://raw.githubusercontent.com/mehmetg06/Remedia/main/runpod/bootstrap.sh | bash
```

Ayrıntılar: [`runpod/README.md`](runpod/README.md)

> RunPod'un gerçek tek-tık deploy URL'si bir RunPod template ID gerektirir.
> Hazır template tanımı `runpod/template.json` içindedir. Template oluşturulduktan
> sonra üstteki düğme `https://console.runpod.io/deploy?template=...` bağlantısına
> çevrilebilir.

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

Colab:

```text
/content/drive/MyDrive/remedia_setup/pocket_cache.json
```

RunPod:

```text
/workspace/remedia_cache/pocket_cache.json
```

Yeni hedef ilk kez kullanıldığında fpocket çalışır. Aynı hedef sonraki
çalıştırmalarda cache'den okunur.

## Notebook kullanımı

### Colab

1. **Runtime → Change runtime type → T4 GPU** seç.
2. Notebook'u `Run all` ile çalıştır.
3. Yeni hedefte Miniconda kernel'i yeniden başlatırsa `Run all`ı tekrar çalıştır.

### RunPod

1. NVIDIA GPU'lu Pod başlat ve JupyterLab'i aç.
2. `Remedia/notebooks/remedia_runpod.ipynb` notebook'unda **Run All Cells** seç.
3. Günlük denemelerde `balanced`, nihai doğrulamada `final` kullan.
4. İşin bitince Pod'u durdur veya sonlandır; sonuçlar `/workspace` volume'unda kalır.

## Repository yapısı

```text
notebooks/remedia_pipeline.ipynb  Colab akışı
notebooks/remedia_runpod.ipynb    RunPod Jupyter akışı
runpod/Dockerfile                 GNINA/fpocket hazır RunPod imajı
runpod/bootstrap.sh               Resmi PyTorch template'i için tek komut kurulum
runpod/template.json              RunPod template ayarları
src/gnina_engine.py               Batch GNINA motoru ve accuracy profilleri
src/molecule_generator.py         Fusion, genetic, BRICS ve random üretim
src/generative_model.py           Opsiyonel REINVENT4 sampling
src/fetch_structure.py            AlphaFold/PDB yapı indirme
src/pocket_detection.py           fpocket entegrasyonu
src/admet_filter.py               Lipinski/Veber drug-likeness filtresi
src/rank_report.py                Docking ve filtre sonuçlarını sıralama
tests/test_gnina_engine.py        GPU gerektirmeyen birim testleri
legacy/                           Eski Streamlit/Snakemake/notebook akışları
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
