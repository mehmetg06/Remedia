# Remedia — Reseptör Odaklı İlaç Keşif Pipeline'ı

Google Colab üzerinde çalışan, reseptör hedefli açık kaynak ilaç keşif prototipi:

**Molekül üretimi → AlphaFold DB → pocket detection → GNINA GPU docking → drug-likeness filtresi → sıralama**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb)

## Hızlandırılmış varsayılan akış

Notebook günlük geliştirme için hafif ayarlarla açılır:

- `GENERATE_N = 10`
- `TOP_FRACTION = 0.10`
- `ACCURACY_PROFILE = "balanced"`
- `INSTALL_REINVENT4 = False`
- `RUN_BENCHMARK = False`

GNINA artık ligand başına ayrı süreç açmaz. Bütün ligandlar FAST aşamasında tek bir çoklu SDF dosyasına yazılır ve **tek GNINA sürecinde** docklanır. Seçilen en iyi adaylar da ACCURATE aşamasında ikinci bir batch süreçte çalışır. Normal iki aşamalı tarama böylece toplamda en fazla iki GNINA süreci kullanır.

Ligandların 3D SDF dosyaları yalnızca bir kez hazırlanır. Accurate aşaması FAST aşamasında hazırlanan aynı konformasyonları yeniden kullanır.

## Accuracy profilleri

| Profil | FAST | ACCURATE | Kullanım |
|---|---|---|---|
| `balanced` | exhaustiveness 4, 1 pose, fast CNN | exhaustiveness 8, 3 pose, varsayılan CNN ensemble | Günlük Colab çalışmaları |
| `final` | exhaustiveness 4, 1 pose, fast CNN | exhaustiveness 16, 9 pose, varsayılan CNN ensemble | Son doğrulama |

`final` profili belirgin biçimde daha yavaştır; yalnızca nihai aday setinde kullanılması önerilir.

## Pocket cache

Pocket merkezi, UniProt ID anahtarıyla şu dosyada saklanır:

```text
/content/drive/MyDrive/remedia_setup/pocket_cache.json
```

Yeni bir hedef ilk kez kullanıldığında fpocket gerekebilir ve Miniconda kernel'i bir kez yeniden başlatabilir. Aynı hedef sonraki çalıştırmalarda cache'den okunur; Miniconda ve fpocket kurulumu tamamen atlanır.

Geometrik merkez fallback sonucu güvenilir bir pocket olmadığı için cache'e yazılmaz.

## Notebook kullanımı

1. Colab'da **Runtime → Change runtime type → T4 GPU** seç.
2. Notebook'u `Run all` ile çalıştır.
3. Yeni hedefte Miniconda kernel'i yeniden başlatırsa `Run all`ı bir kez daha çalıştır.
4. Günlük denemelerde `balanced`, nihai doğrulamada `final` kullan.
5. REINVENT4 yalnızca `pretrained` yöntemi seçilecekse açılmalıdır.
6. Benchmark yalnızca hız/skor karşılaştırması gerektiğinde açılmalıdır.

## Repository yapısı

```text
notebooks/remedia_pipeline.ipynb  Ana Colab akışı
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

Testler balanced/final profil bayraklarını, top %10 seçimini, stale-output korumasını, tek modda bir batch çağrısını ve iki aşamalı akışta toplam iki batch çağrısını doğrular.

## Bilimsel kullanım notu

Remedia bir araştırma ve ön eleme prototipidir. Docking skoru tek başına bağlanma veya etkinlik kanıtı değildir. Son adaylar deneysel yapı, reseptör hazırlama, uygun kontroller ve laboratuvar doğrulamasıyla değerlendirilmelidir.

## Lisans

AGPL-3.0-or-later. Ayrıntılar için [LICENSE](LICENSE).
