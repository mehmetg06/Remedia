# Remedia — Reseptör Odaklı İlaç Keşif Pipeline'ı

Açık kaynak, reseptör hedefli (structure-based) ilaç molekülü keşif pipeline'ı —
**tamamen tek bir Google Colab notebook'unda**, baştan sona çalışır.

Molekül Üretimi → AlphaFold DB → Pocket Detection → **GNINA (GPU) Docking** →
ADMET Filtreleme → Sıralama → Görsel Sonuç.

---

## 🚀 Nasıl çalıştırılır (tek yol, 5 adım)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb)

1. Yukarıdaki **“Open in Colab”** rozetine tıkla — notebook Colab'da açılır.
2. **Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ GPU (T4)** seç.
3. **Runtime ▸ Run all**. İlk **Hücre 0** (Miniconda kurulumu) **kernel'i yeniden
   başlatır** — bu NORMAL, endişelenme.
4. Kernel yeniden başladıktan sonra **`Run all`'ı TEKRAR çalıştır**; bu sefer
   Miniconda zaten kurulu olduğu için restart olmadan baştan sona akar.
5. En alttaki hücrede sonuç tablosunu, molekül çizimlerini ve Google Drive'a
   kaydedilen dosyaları gör.

Hepsi bu kadar. **Hiç dosya taşıma, hiç git senkronizasyonu, hiç kopyala-yapıştır
yok** — her şey Colab'da olur.

> **Parametreler kod içine elle yazılmaz:** UniProt ID, üretim yöntemi
> (füzyon/genetik/BRICS/random), molekül sayısı gibi seçimler ilgili hücrelerin
> üstündeki **Colab form kutularından** (dropdown/kaydırıcı) yapılır. Yalnızca
> çok satırlı SMILES tohum listesi kod içindeki üçlü-tırnaklı `MANUAL_SEEDS`
> değişkenine yapıştırılır (form kutuları tek satırlıktır).

> **Not:** Notebook fpocket'i conda ile kurar ve gerekli **conda Terms of
> Service'i otomatik kabul eder**.

> Doğrudan bağlantı:
> `https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb`

## ⚠️ GPU ZORUNLUDUR

Docking motoru **GNINA**'dır ve **GPU'da** çalışır (Colab'ın ücretsiz T4'ü yeter).
GPU seçmeden **Hücre 5 (docking) çalışmaz**. Notebook'u açar açmaz
**Runtime ▸ Change runtime type ▸ GPU (T4)** yapmayı unutma. AutoDock Vina
tamamen bırakılmıştır.

## 🧬 Notebook ne yapıyor? (Hücre 0 + 8 hücre)

Notebook yukarıdan aşağıya çalıştırılır; her hücrenin üstünde ne yaptığını, ne
kadar süreceğini ve devam etmeden önce neyi görmen gerektiğini yazan bir not var.

| # | Hücre | Ne yapar |
|---|---|---|
| 0 | **Miniconda** | fpocket'i conda ile kurabilmek için `condacolab` ile Miniconda kurar. **Kernel'i yeniden başlatır** — restart sonrası `Run all`'ı tekrar çalıştır. İkinci turda zaten kurulu olduğu için restart olmaz. |
| 1 | **Kurulum** | GNINA (GPU binary), **fpocket'i conda ile** (conda ToS'u otomatik kabul ederek), RDKit, meeko + Python paketleri kurar; `src/`'yi import yoluna ekler; GPU'yu kontrol eder. |
| 2 | **Hedef** | `UNIPROT_ID` (form kutusundan; varsayılan `P00918`, Karbonik Anhidraz II) için AlphaFold yapısını **REST API'den** indirir; fpocket ile en druggable cebi bulup merkezini hesaplar (fpocket yoksa geometrik merkeze düşer). |
| 3 | **Tohum moleküller** | `known_ligands.py` ile ChEMBL/PubChem'den bilinen inhibitörleri çeker; bulamazsa `MANUAL_SEEDS` (üçlü-tırnaklı SMILES metni) kullanılır. |
| 3.5 | **REINVENT4 kurulumu (opsiyonel)** | `generative_model.py`, [REINVENT4](https://github.com/MolecularAI/REINVENT4)'ü klonlar/kurar ve halka açık, önceden eğitilmiş prior ağırlığını indirir — yalnızca `pretrained` yöntemini kullanacaksan gerekir; TEK SEFERLİK ve atlanabilir (form kutusu). |
| 4 | **Molekül üret** | Yeni aday moleküller üretir; yöntem **form kutusundan** seçilir: füzyon / genetik / BRICS / random (`molecule_generator.py`, tohum gerektirir) veya **pretrained** (`generative_model.py`, tohum GEREKTİRMEZ). |
| 5 | **GNINA Docking (GPU)** | `gnina_engine.py` ile iki-aşamalı docking: TÜM adaylar önce **FAST** modda hızlıca elenir, en iyi top-N/top-% **ACCURATE** modda yeniden docklanır (form kutusundan `sadece_fast`/`sadece_accurate` de seçilebilir). Nihai skorlar ACCURATE'ten gelir; `ligand, affinity_kcal_mol, skor_kaynagi, ...` DataFrame'i üretir. |
| 5.5 | **(Opsiyonel) Benchmark** | `gnina_engine.benchmark_fast_vs_accurate` ile aynı molekülleri hem FAST hem ACCURATE dockleyip gerçek süre/skor farkını ölçer — atlanabilir. |
| 6 | **ADMET** | `admet_filter.py` ile Lipinski/Veber filtresi uygular. |
| 7 | **Sırala** | `rank_report.py` ile docking + ADMET'i birleşik sıralar. |
| 8 | **Sonuç** | En iyi adayları tablo + RDKit 2D çizimlerle gösterir; tüm sonuçları Google Drive'a tarihli klasöre kaydeder (kalıcılık). |

## 📦 Bu repoda ne var?

```
notebooks/remedia_pipeline.ipynb   ← ANA VE TEK AKIŞ (Colab, GPU)
src/                               ← notebook'un import ettiği çekirdek modüller
  fetch_structure.py       AlphaFold/PDB'den yapı indirme (REST API)
  pocket_detection.py      fpocket ile bağlanma cebi tespiti
  known_ligands.py         ChEMBL/PubChem'den bilinen ligandlar
  molecule_generator.py    kural tabanlı yeni molekül üretimi (füzyon/GA)
  generative_model.py      REINVENT4 (önceden eğitilmiş prior) ile üretim
  ligand_prep.py           SMILES → 3D konformasyon
  gnina_engine.py          GNINA docking motoru — fast/accurate mod + iki-aşamalı pipeline
  admet_filter.py          Lipinski/Veber ADMET filtresi
  rank_report.py           docking + ADMET birleşik sıralaması
data/                              ← örnek girdi molekülleri
tests/                             ← birim testleri (python -m unittest discover tests)
legacy/                            ← eski/opsiyonel arayüzler (aşağıya bak)
```

## 🗂️ Eski arayüzler (`legacy/`) — opsiyonel

Önceki Streamlit UI, Snakemake akışı, Codespaces devcontainer'ı ve eski Colab
notebook'ları **artık ana akış değildir**; [`legacy/`](legacy/) klasörüne taşındı.
Silinmediler — referans için oradalar ama aktif bakımı yapılmıyor. Ayrıntı:
[`legacy/README.md`](legacy/README.md).

## 🧪 Molekül üretim yöntemleri (Hücre 4)

| Yöntem | Tohum gerekli mi? | Nasıl üretir? |
|---|---|---|
| **fusion** (varsayılan) | Evet | Geniş keşif → ön eleme → genetik optimizasyon → rafinasyon (`molecule_generator.py`, RDKit kural tabanlı). |
| **genetic** | Evet | Saf genetik algoritma. |
| **brics** | Evet | BRICS fragman rekombinasyonu. |
| **random** | Evet | Rastgele atom/grup mutasyonu. |
| **pretrained (REINVENT4)** | **Hayır** | Tohum molekül gerektirmeden, önceden eğitilmiş bir yapay zeka modeliyle (RNN tabanlı [REINVENT4](https://github.com/MolecularAI/REINVENT4) "prior"ı) sıfırdan ilaç-benzeri molekül üretir. **Reseptöre özel değildir** — model hiçbir reseptörü "bilmez", yalnızca genel kimyasal olarak makul, çeşitli moleküller üretir; üretilen moleküller sonradan GNINA docking + ADMET ile test edilir. Reseptöre özel eğitim (fine-tuning/RL) **yapılmaz**. |

`pretrained` seçilirse Hücre 3.5 (veya Hücre 4'ün kendisi, ilk kullanımda) REINVENT4'ü
GitHub'dan kurar ve AstraZeneca'nın Zenodo'da yayınladığı halka açık prior ağırlığını
indirir — ~3-5 GB indirme, ilk seferde birkaç dakika sürer, sonrasında atlanır.

## 🎯 Farklı bir hedef denemek

Notebook'un **Hücre 2**'sinde `UNIPROT_ID`'yi değiştir (ör. `P30405` — CypD).
Yapı, cep merkezi ve bilinen ligandlar o hedefe göre otomatik güncellenir. Başka
hiçbir şeye dokunman gerekmez.

## 📄 Lisans

AGPL-3.0 — özgürce fork'la, katkı ver; ancak türev çalışmalar (ağ üzerinden
sunulan servisler dahil) de açık kaynak kalmak zorunda. Bkz. [LICENSE](LICENSE).
