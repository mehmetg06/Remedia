# Remedia — Reseptör Odaklı İlaç Keşif Pipeline'ı

Açık kaynak, reseptör hedefli (structure-based) ilaç molekülü keşif pipeline'ı —
**tamamen tek bir Google Colab notebook'unda**, baştan sona çalışır.

Molekül Üretimi → AlphaFold DB → Pocket Detection → **GNINA (GPU) Docking** →
ADMET Filtreleme → Sıralama → Görsel Sonuç.

---

## 🚀 Nasıl çalıştırılır (tek yol, 4 adım)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb)

1. Yukarıdaki **“Open in Colab”** rozetine tıkla — notebook Colab'da açılır.
2. **Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ GPU (T4)** seç.
3. **Runtime ▸ Run all**.
4. En alttaki hücrede sonuç tablosunu, molekül çizimlerini ve Google Drive'a
   kaydedilen dosyaları gör.

Hepsi bu kadar. **Hiç dosya taşıma, hiç git senkronizasyonu, hiç kopyala-yapıştır
yok** — her şey Colab'da olur.

> Doğrudan bağlantı:
> `https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/remedia_pipeline.ipynb`

## ⚠️ GPU ZORUNLUDUR

Docking motoru **GNINA**'dır ve **GPU'da** çalışır (Colab'ın ücretsiz T4'ü yeter).
GPU seçmeden **Hücre 5 (docking) çalışmaz**. Notebook'u açar açmaz
**Runtime ▸ Change runtime type ▸ GPU (T4)** yapmayı unutma. AutoDock Vina
tamamen bırakılmıştır.

## 🧬 Notebook ne yapıyor? (8 hücre)

Notebook yukarıdan aşağıya çalıştırılır; her hücrenin üstünde ne yaptığını, ne
kadar süreceğini ve devam etmeden önce neyi görmen gerektiğini yazan bir not var.

| # | Hücre | Ne yapar |
|---|---|---|
| 1 | **Kurulum** | GNINA (GPU binary), fpocket, RDKit, meeko + Python paketleri kurar; `src/`'yi import yoluna ekler; GPU'yu kontrol eder. |
| 2 | **Hedef** | `UNIPROT_ID` (varsayılan `P00918`, Karbonik Anhidraz II) için AlphaFold yapısını **REST API'den** indirir; fpocket ile en druggable cebi bulup merkezini hesaplar. |
| 3 | **Tohum moleküller** | `known_ligands.py` ile ChEMBL/PubChem'den bilinen inhibitörleri çeker; bulamazsa `MANUAL_SEEDS` (üçlü-tırnaklı SMILES metni) kullanılır. |
| 4 | **Molekül üret** | `molecule_generator.py` füzyon motoruyla yeni aday moleküller üretir. |
| 5 | **GNINA Docking (GPU)** | Her molekülü 3D'ye hazırlayıp GNINA (CNN rescoring) ile dockler; `ligand, affinity_kcal_mol` DataFrame'i üretir. |
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
  ligand_prep.py           SMILES → 3D konformasyon
  admet_filter.py          Lipinski/Veber ADMET filtresi
  rank_report.py           docking + ADMET birleşik sıralaması
data/                              ← örnek girdi molekülleri
legacy/                            ← eski/opsiyonel arayüzler (aşağıya bak)
```

## 🗂️ Eski arayüzler (`legacy/`) — opsiyonel

Önceki Streamlit UI, Snakemake akışı, Codespaces devcontainer'ı ve eski Colab
notebook'ları **artık ana akış değildir**; [`legacy/`](legacy/) klasörüne taşındı.
Silinmediler — referans için oradalar ama aktif bakımı yapılmıyor. Ayrıntı:
[`legacy/README.md`](legacy/README.md).

## 🎯 Farklı bir hedef denemek

Notebook'un **Hücre 2**'sinde `UNIPROT_ID`'yi değiştir (ör. `P30405` — CypD).
Yapı, cep merkezi ve bilinen ligandlar o hedefe göre otomatik güncellenir. Başka
hiçbir şeye dokunman gerekmez.

## 📄 Lisans

AGPL-3.0 — özgürce fork'la, katkı ver; ancak türev çalışmalar (ağ üzerinden
sunulan servisler dahil) de açık kaynak kalmak zorunda. Bkz. [LICENSE](LICENSE).
