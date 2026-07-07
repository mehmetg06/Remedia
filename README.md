# Receptor-Focused Drug Discovery Pipeline

Açık kaynak, reseptör hedefli (structure-based) ilaç molekülü keşif pipeline'ı.
Molekül Üretimi → AlphaFold DB → Pocket Detection → Docking (**GNINA · Colab GPU**) → ADMET Filtreleme

> **Docking motoru GNINA'dır.** AutoDock Vina tamamen bırakıldı. Docking GPU'da,
> ücretsiz Google Colab T4'ünde (`notebooks/gnina_colab.ipynb`) çalışır; sonuç
> `docking_scores.csv` olarak Remedia'ya yüklenir ve pipeline aynen devam eder.
> Ayrıntı: aşağıdaki [GNINA ile Docking (Colab, GPU)](#gnina-ile-docking-colab-gpu) bölümü.

## Mimari

```
        (opsiyonel) molecule_generator.py → sıfırdan YENİ molekül üret
                    │  (random / BRICS / genetik algoritma)
                    ▼
Reseptör (UniProt ID / PDB ID)
        │
        ▼
1. fetch_structure.py     → AlphaFold/PDB'den 3D yapı indir
        │
        ▼
2. pocket_detection.py    → Binding site (cep) tespiti
        │
        ▼
3. ligand_prep.py         → Aday moleküller (SMILES listesi / generated.smi)
        │
        ▼
4. GNINA (Colab, GPU)     → notebooks/gnina_colab.ipynb ile skorlama
   docking.py              → Colab çıktısı docking_scores.csv'yi doğrular
        │
        ▼
5. admet_filter.py        → ADMETlab 3.0 API ile farmakokinetik eleme
        │
        ▼
6. rank_report.py         → Birleşik skor + rapor (CSV/HTML)
```

Tüm bunları tek bir web arayüzünden yönetmek için: **`streamlit run app.py`**
(aşağıya bkz.).

## Kurulum (GitHub Codespaces / Linux)

Yeni bir Codespace açıldığında `setup.sh` **otomatik** çalışır (`.devcontainer/devcontainer.json`
içindeki `postCreateCommand` sayesinde) ve gerekli tüm araçları (Python paketleri, openbabel,
conda/Miniconda, fpocket) tek seferde kurar. İlk kurulum birkaç dakika sürebilir —
özellikle Miniconda indirmesi yüzünden.

> **Not:** Docking için Codespaces'e yerel bir docking motoru kurmana gerek yok —
> docking GPU'da, Colab'da GNINA ile yapılır (aşağıya bkz.). Codespaces yalnızca
> molekül üretimi, ADMET, sıralama ve dashboard için kullanılır.

Eski/mevcut bir Codespace'te bu otomasyon yoksa (ör. `.devcontainer` güncellenmeden önce açılmış
bir ortam), elle çalıştırman yeterli:

```bash
bash setup.sh
```

Script idempotent'tir: zaten kurulu olan araçları atlar, zarar vermeden defalarca çalıştırılabilir.
Sonunda hangi araçların kurulu olduğunu gösteren bir özet tablo (✓/✗) yazdırır.

## Hızlı Başlangıç — Web Arayüzü (UI)

En kolay yol: tek komutla açılan Streamlit arayüzü. Codespaces'te de çalışır.

```bash
streamlit run app.py
```

Arayüz seni 5 adımda yönlendirir: **(1)** hedef protein seçimi → **(2)** bağlanma
cebi → **(3)** molekül üretim yöntemi → **(4)** çalıştırma (canlı log) → **(5)**
sonuçlar (2D çizim + skorlar + sade yorum + ham CSV indirme).

Tasarım prensibi: **basit/gelişmiş diye ayrı mod yok**. Her ekranda ham teknik
değer (ör. `pLDDT: 88.9`) İLE sade açıklaması (ör. "yapının ne kadar güvenilir
tahmin edildiğini gösterir") yan yana, hep birlikte görünür. Teknik kullanıcı
ham sayıları görür; hiç bilmeyen kullanıcı yanındaki cümleden anlar.

## Yeni Molekül Üretimi (`molecule_generator.py`)

Var olan molekülleri test etmenin ötesinde, **sıfırdan yeni aday molekül üretir** —
model eğitmeden, yalnızca RDKit ile kimyasal kurallara dayanarak. Dört yöntem:

| Yöntem | Ne yapar | Model gerekir mi? |
|--------|----------|-------------------|
| **random** | Tohum moleküle rastgele atom değişimi (C↔N↔O↔S) ve grup ekleme/çıkarma (metil, hidroksil, halojen); geçersizleri RDKit ile eler | Hayır |
| **brics** | Moleküllerin fragmanlarını BRICS kurallarıyla söküp LEGO gibi yeniden birleştirir | Hayır |
| **genetic** | Genetik algoritma: her nesilde docking ile skorlar (fitness = −affinity), en iyi %20'yi tutar, çaprazlama + mutasyonla yeniler | Hayır |
| **pretrained** | REINVENT gibi HAZIR bir model için opsiyonel plugin arayüzü (stub) | Evet (opsiyonel) |

Çıktı, mevcut pipeline'a **doğrudan** giren bir `.smi` dosyasıdır.

```bash
# Rastgele mutasyon
python src/molecule_generator.py --method random \
    --seeds "CC(=O)Oc1ccccc1C(=O)O" --n 50 --output data/generated.smi

# BRICS rekombinasyonu (birden fazla tohum)
python src/molecule_generator.py --method brics \
    --seeds-file data/ligands_example.smi --n 50 --output data/generated.smi

# Genetik algoritma (reseptör verilirse gerçek Vina docking; yoksa QED yedek fitness)
python src/molecule_generator.py --method genetic \
    --seeds-file data/ligands_example.smi --generations 10 --population 30 \
    --receptor data/P30405_alphafold.pdbqt --center 5.00 -1.02 -15.56 \
    --size 20 20 20 --output data/generated.smi
```

### Neden Doğrulama Gerekiyor?

Düşük `exhaustiveness` (ör. 8, varsayılan) Vina'nın arama uzayını yeterince
kapsamlı taramasını engelleyebilir. Genetik algoritma bu koşulda bazen "şanslı"
ama gerçekçi olmayan bir bağlanma pozu bulur ve çok iyi görünen bir skor raporlar
(örn. **−11.6 kcal/mol**). Aynı molekülü `exhaustiveness=32` ile yeniden
docklandığında skor **−5.2 kcal/mol**'e düşebilir — aralarındaki 6.4 kcal/mol'lük
fark, birinci sonucun bir **arama artefaktı** olduğunu gösterir, gerçek bir
bağlanma gücünü değil.

> **⚠️ ÖNEMLİ NOT:** Genetik Algoritma (GA), skoru doğrudan maksimize etmeye çalıştığından, düşük arama yoğunluğundaki (exhaustiveness) bu "şişirilmiş artefakt skorları" bulup sömürmeye (exploit) aşırı yatkındır. Bu yüzden GA sonrasında elde edilen en iyi adayların yüksek `exhaustiveness` (ör. 32) ve farklı motorlarla (Smina/Vina çapraz doğrulama) **MUTLAKA** teyit edilmesi gerekir. Aksi takdirde GA, aslında iyi bağlanmayan ama şans eseri iyi skorlanmış "yalancı pozitif" (false positive) moleküllere erken yakınsayacaktır.

Bu yüzden `validate_top_candidates.py` adımı pipeline'a eklendi:
en iyi N aday otomatik olarak yüksek exhaustiveness ile yeniden docklanır ve
her molekül için **GÜVENİLİR / ŞÜPHELİ / ARTEFAKT OLASI** etiketi üretilir.

```bash
python src/validate_top_candidates.py \
    --input results/docking_scores.csv \
    --receptor data/P30405_alphafold.pdbqt \
    --center 5.00 -1.02 -15.56 --size 20 20 20 \
    --top-n 5 --exhaustiveness 32 \
    --output results/validated_candidates.csv
```

Çıktı CSV sütunları: `ligand`, `ilk_skor`, `dogrulanmis_skor`, `fark`, `guven_durumu`.

Üretilen molekülleri tam pipeline'a sokmak (Snakemake ile):

```bash
snakemake --cores 1 generate                                 # data/generated.smi üret
snakemake --cores 1 --config ligands_file=data/generated.smi # dock + ADMET + sırala
```

## GNINA ile Docking (Colab, GPU)

Docking motoru **GNINA**'dır (GPU'da çalışan derin öğrenmeli docking + CNN
rescoring). Vina tamamen bırakıldı. Docking, Codespaces yerine ücretsiz **Google
Colab T4 GPU**'sunda yapılır; sonuç `docking_scores.csv` olarak geri yüklenir.
Codespaces'te GPU gerekmez.

Teknik bilgi gerekmez — notebook her adımda ne yapacağını sana söyler. Sırayla:

1. Remedia UI'da (`streamlit run app.py`) **Adım 4**'te molekülleri üret, **Adım 5**'te
   **"💾 Molekülleri Kaydet & Docking'e Hazırla"** butonuna bas. Sana bir
   `run_id` verir (ör. `run_20260707_101500_a3f2`) ve `results/<run_id>/` klasörünü açar.
2. **"Open in Colab"** linkine tıkla:
   👉 https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/gnina_colab.ipynb
3. Colab'da üstten **Runtime ▸ Change runtime type ▸ GPU (T4)** seç.
4. Tüm hücreleri **yukarıdan aşağıya SIRAYLA** çalıştır (`Shift+Enter`). Bir sonrakine
   geçmeden önce her hücrenin çıktısında **`✅`** işaretini gör.
   - **ADIM 3**'te reseptörü (UniProt ID / dosya), pocket merkezini ve molekülleri
     kutucuklardan seç. Molekülleri `Remedia/data/generated.smi`'den okur ya da
     `MANUAL_SMILES` kutusuna elle yapıştırabilirsin.
5. Son hücre otomatik olarak **`docking_scores.csv`** dosyasını bilgisayarına indirir.
6. Bu dosyayı Codespaces'te **`results/<run_id>/`** klasörüne (adım 1'deki `run_id`)
   `docking_scores.csv` adıyla sürükle-bırak ile yükle (üzerine yaz).
7. Remedia UI'da **"✅ Docking Tamamlandı, Devam Et"** butonuna bas. Dosya bulununca
   pipeline'ın kalanı (ADMET → sıralama → dashboard) **otomatik** çalışır — yeni bir
   script çalıştırmana gerek yok, format birebir aynıdır.

> **CLI tercih edersen:** Colab'dan inen dosyayı `results/<run_id>/docking_scores.csv`
> olarak koy, sonra `snakemake --cores 1 --config ligands_file=data/generated.smi run_id=<run_id>`
> çalıştır. Dosya mevcut olduğundan docking adımı atlanır, kalan adımlar koşar.

**Çıktı formatı (ZORUNLU) —** dosya tam olarak şu iki sütunla, BAŞKA HİÇBİR EK
SÜTUN OLMADAN olmalı:

```csv
ligand,affinity_kcal_mol
mol_1,-8.4
mol_2,-7.1
```

Bu format mevcut `admet_filter.py` ve `rank_report.py` tarafından doğrudan okunur.
Doğrulamak için: `python src/docking.py --validate-only results/<run_id>/docking_scores.csv`

### Sorun mu yaşıyorsun?

En sık karşılaşılan 3 durum ve NET çözümleri:

- **"⚠️ GPU bulunamadı!"** (ADIM 1'de)
  Colab sana GPU vermemiş. Üstten **Runtime ▸ Change runtime type ▸ Hardware
  accelerator ▸ GPU (T4)** seç, **Save**'e bas, sonra ADIM 1 hücresini **tekrar**
  çalıştır. Ücretsiz T4 kotan dolduysa birkaç saat sonra tekrar dene.

- **"GNINA Colab'da kurulamadı"** (ADIM 2'de)
  Neredeyse her zaman geçici bir indirme/erişim sorunudur — GNINA binary'si büyüktür.
  Çözüm: **ADIM 2 hücresini tekrar çalıştır** (indirme yarıda kesildiyse baştan alır).
  Hücre, en güncel sürüm inmezse otomatik olarak bilinen kararlı sürümlere (v1.3 →
  v1.1 → v1.0.3) düşer. Yine de olmazsa **Runtime ▸ Restart runtime** yapıp ADIM 2'yi
  baştan çalıştır. `gnina --version` çıktısı hâlâ gelmiyorsa GPU runtime seçili
  olmayabilir; önce ADIM 1'e dön.

- **"docking_scores.csv formatı uyuşmuyor"** (Codespaces'te "Devam Et" hatası)
  Dosya mutlaka **`ligand,affinity_kcal_mol`** başlığıyla ve **başka sütun olmadan**
  kaydedilmeli. Notebook'un ADIM 5'i bunu otomatik doğru yazar; dosyayı elle
  düzenlediysen fazladan sütunları sil. Ayrıca dosyanın doğru klasörde olduğundan
  emin ol: tam olarak `results/<run_id>/docking_scores.csv` (İndirilenler klasöründe
  kalmış olabilir). Kontrol: `python src/docking.py --validate-only results/<run_id>/docking_scores.csv`

## DiffDock ile GPU Doğrulama (Colab)

Vina fiziksel bir skorlama yapar; **DiffDock** ise derin öğrenmeyle bağlanma
pozunu tahmin edip bağımsız bir **güven skoru** verir. İki farklı yöntem de bir
molekülü güçlü buluyorsa ona daha çok güvenebilirsin. DiffDock GPU ister; bu yüzden
onu Google Colab'ın **ücretsiz T4 GPU**'sunda çalıştırıp sonucu Codespaces'e geri
getiriyoruz. Codespaces'te GPU'ya gerek yok.

Tek yapman gereken, aşağıdaki numaralı adımları sırayla izlemek — teknik bilgi
gerekmez, notebook her adımda ne yapacağını sana söyler:

1. **`notebooks/diffdock_colab.ipynb`** dosyasını GitHub'da aç.
2. **"Open in Colab"** linkine tıkla:
   👉 https://colab.research.google.com/github/mehmetg06/Remedia/blob/main/notebooks/diffdock_colab.ipynb
3. Colab açılınca üstten **Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ GPU (T4)** seç.
4. Yukarıdan aşağıya **HER HÜCREYİ SIRAYLA** çalıştır (`Shift+Enter`). Bir sonrakine
   geçmeden önce her hücrenin çıktısında **`✅`** işaretini gör.
5. Son hücre otomatik olarak **`diffdock_results.csv`** dosyasını bilgisayarına indirir.
6. Bu dosyayı Codespaces'teki **`Remedia/results/`** klasörüne sürükle-bırak ile yükle.
7. Codespaces terminalinde şunu çalıştır:
   ```bash
   python src/merge_diffdock_results.py
   ```
8. Çıktıda **"GÜÇLÜ ADAY"** olarak işaretlenen moleküller en güvenilir sonuçlarındır
   (hem Vina hem DiffDock güçlü bulmuştur). Detaylı tablo: `results/final_comparison.csv`.

Birleştirme mantığı — `genel_guven_durumu` sütunu:

| Durum | Anlamı |
|-------|--------|
| **GÜÇLÜ ADAY** | Hem Vina (≤ −7.0 kcal/mol) hem DiffDock (güven ≥ 0) güçlü buldu |
| **TEK YÖNTEMLE DESTEKLENİYOR** | Sadece bir yöntem güçlü buldu — temkinli ol |
| **ZAYIF ADAY** | İkisi de zayıf — muhtemelen elenmeli |

> **Not:** `merge_diffdock_results.py`, `results/validated_candidates.csv` dosyasındaki
> `dogrulanmis_skor`u (yoksa `ilk_skor`) `vina_affinity` olarak kullanır. Vina dosyası
> yoksa script kırılmaz; sadece DiffDock skorlarını raporlar.

### Sorun mu yaşıyorsun?

En sık karşılaşılan 3 durum ve NET çözümleri:

- **"⚠️ GPU bulunamadı!"** (ADIM 1'de)
  Colab sana GPU vermemiş. Üstten **Runtime ▸ Change runtime type ▸ Hardware
  accelerator ▸ GPU (T4)** seç, **Save**'e bas, sonra ADIM 1 hücresini **tekrar**
  çalıştır. Ücretsiz T4 kotan dolduysa birkaç saat sonra tekrar dene.

- **"torch_geometric / torch_scatter kurulum hatası"** (ADIM 2'de)
  Neredeyse her zaman torch↔CUDA sürüm uyuşmazlığındandır. ADIM 2 hücresi, PyG
  tekerleklerini (`torch_scatter`, `torch_sparse`, `torch_cluster`) senin torch
  sürümüne göre otomatik seçer. Yine de patlarsa: **Runtime ▸ Restart runtime** yap
  ve ADIM 2'yi baştan çalıştır (bazı paketler ancak yeniden başlatınca doğru yüklenir).
  Hâlâ olmuyorsa ADIM 2'nin başındaki `torch=... cuda=...` çıktısına bak — CUDA `cpu`
  görünüyorsa GPU runtime seçili değildir; önce ADIM 1'e dön.

- **"diffdock_results.csv bulunamadı"** (Codespaces'te `merge_diffdock_results.py`)
  Colab'dan indirdiğin dosyayı **doğru klasöre** koymadın. Dosyanın tam olarak
  `Remedia/results/diffdock_results.csv` yolunda olduğundan emin ol (İndirilenler
  klasöründe kalmış olabilir). Dosyayı VS Code / Codespaces'te `results/` klasörüne
  sürükle-bırak ile yükle, sonra komutu tekrar çalıştır.

## Hızlı Başlangıç — Tek Komut (CLI)

Tüm pipeline artık **Snakemake** ile otomatik çalışıyor. Parametreler
`config.yaml`'dan okunur; tek yapman gereken:

```bash
snakemake --cores 1
```

Bu komut sırayla ve **sadece eksik/değişen adımları** çalıştırır:

1. Yapı yoksa AlphaFold DB'den indirir (`fetch_structure.py`)
2. Binding pocket'ı `config.yaml`'daki merkezle sabitler (`pocket_detection.py`)
3. Ligandları hazırlar: SMILES → 3D (`ligand_prep.py`)
4. **Docking:** `results/<run_id>/docking_scores.csv` VARSA doğrulayıp kullanır
   (`docking.py --validate-only`); YOKSA seni Colab'da GNINA çalıştırmaya yönlendirip
   burada durur. (Docking artık GPU'da, Colab'da GNINA ile yapılır.)
5. ADMET filtresi uygular (`admet_filter.py`)
6. Nihai sıralamayı üretir (`rank_report.py`)
7. `dashboard.html`'i güncel sonuçlarla yeniden yazar (`generate_dashboard.py`)

Yani `snakemake --cores 1` ilk çalıştırmada docking adımında **bilerek durur** ve
sana Colab talimatını verir; `docking_scores.csv`'yi `results/<run_id>/` klasörüne
yükleyip komutu tekrar çalıştırınca kalan adımlar otomatik koşar. Snakemake
input/output dosyalarını izlediği için indirme/üretim adımları tekrar çalışmaz (caching).

### Parametreleri değiştirme (`config.yaml`)

Farklı bir reseptör, cep merkezi veya ligand listesi için `config.yaml`'ı
düzenle — komut satırında bir şey yazmana gerek yok:

```yaml
uniprot_id: "P30405"                 # AlphaFold DB UniProt ID (örn. CypD/PPIF)
pocket_center: [5.00, -1.02, -15.56] # docking box merkezi (x, y, z)
box_size: [20.0, 20.0, 20.0]         # kutu boyutu (Å)
ligands_file: "data/ligands_example.smi"
exhaustiveness: 8                    # (eski Vina parametresi — GNINA akışında kullanılmaz)
admet_mode: "lipinski"              # veya "admetlab"
```

Tek seferlik override de mümkün:

```bash
snakemake --cores 1 --config exhaustiveness=16
```

Faydalı komutlar:

```bash
snakemake -n            # dry-run: hangi adımların çalışacağını göster
snakemake --dag | dot -Tsvg > dag.svg   # bağımlılık grafiği
snakemake --forceall    # her şeyi baştan çalıştır
```

## Manuel çalıştırma (debug için)

Snakemake akışı önerilir. Tek tek adımları elle çalıştırmak istersen
(hata ayıklama, ara çıktı inceleme) script'ler bağımsız da çalışır:

```bash
python src/fetch_structure.py --uniprot P30405            # → data/P30405_alphafold.pdb
python src/pocket_detection.py --pdb data/P30405_alphafold.pdb \
    --center 5.00 -1.02 -15.56 --size 20
python src/ligand_prep.py --input data/ligands_example.smi \
    --output data/ligands_prepared
# Docking: Colab'da GNINA çalıştır (notebooks/gnina_colab.ipynb), inen
# docking_scores.csv'yi results/<run_id>/ altına koy, sonra formatını doğrula:
python src/docking.py --validate-only results/<run_id>/docking_scores.csv
python src/admet_filter.py --smiles-file data/ligands_example.smi \
    --mode lipinski --output results/admet_results.csv
python src/rank_report.py --docking results/docking_scores.csv \
    --admet results/admet_results.csv --output results/final_ranking.csv
python src/generate_dashboard.py --ranking results/final_ranking.csv \
    --admet results/admet_results.csv --config config.yaml --output dashboard.html
```

## Durum

🚧 İskelet aşaması — her modül minimal çalışır halde, geliştirilmeye açık.

## Lisans

AGPL-3.0 — özgürce fork'la, katkı ver; ancak türev çalışmalar (ağ üzerinden
sunulan servisler dahil) de açık kaynak kalmak zorunda. Bkz. [LICENSE](LICENSE).
