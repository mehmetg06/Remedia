# Receptor-Focused Drug Discovery Pipeline

Açık kaynak, reseptör hedefli (structure-based) ilaç molekülü keşif pipeline'ı.
Molekül Üretimi → AlphaFold DB → Pocket Detection → Docking (Vina) → ADMET Filtreleme

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
4. docking.py             → AutoDock Vina ile skorlama
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
conda/Miniconda, fpocket, vina, smina) tek seferde kurar. İlk kurulum birkaç dakika sürebilir —
özellikle Miniconda indirmesi yüzünden.

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

## Hızlı Başlangıç — Tek Komut (CLI)

Tüm pipeline artık **Snakemake** ile otomatik çalışıyor. Parametreler
`config.yaml`'dan okunur; tek yapman gereken:

```bash
snakemake --cores 1
```

Bu komut sırayla ve **sadece eksik/değişen adımları** çalıştırır:

1. Yapı yoksa AlphaFold DB'den indirir (`fetch_structure.py`)
2. Reseptörü docking için PDBQT'ye çevirir (openbabel)
3. Binding pocket'ı `config.yaml`'daki merkezle sabitler (`pocket_detection.py`)
4. Ligandları hazırlar: SMILES → 3D → PDBQT (`ligand_prep.py`)
5. AutoDock Vina ile dockler (`docking.py`)
6. ADMET filtresi uygular (`admet_filter.py`)
7. Nihai sıralamayı üretir (`rank_report.py`)
8. `dashboard.html`'i güncel sonuçlarla yeniden yazar (`generate_dashboard.py`)

Snakemake input/output dosyalarını izlediği için, örneğin sadece ligand
listesini değiştirirsen yapı indirme adımı tekrar çalışmaz (caching).

### Parametreleri değiştirme (`config.yaml`)

Farklı bir reseptör, cep merkezi veya ligand listesi için `config.yaml`'ı
düzenle — komut satırında bir şey yazmana gerek yok:

```yaml
uniprot_id: "P30405"                 # AlphaFold DB UniProt ID (örn. CypD/PPIF)
pocket_center: [5.00, -1.02, -15.56] # docking box merkezi (x, y, z)
box_size: [20.0, 20.0, 20.0]         # kutu boyutu (Å)
ligands_file: "data/ligands_example.smi"
exhaustiveness: 8                    # Vina arama yoğunluğu
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
obabel data/P30405_alphafold.pdb -O data/P30405_alphafold.pdbqt -xr
python src/pocket_detection.py --pdb data/P30405_alphafold.pdb \
    --center 5.00 -1.02 -15.56 --size 20
python src/ligand_prep.py --input data/ligands_example.smi \
    --output data/ligands_prepared
python src/docking.py --receptor data/P30405_alphafold.pdbqt \
    --ligands-dir data/ligands_prepared \
    --center 5.00 -1.02 -15.56 --size 20 20 20 \
    --output results/docking_scores.csv
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
