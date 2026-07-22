# Remedia on Modal

## Önerilen yol: bir kez deploy et, sabit URL kendini günceller

Bu yol notebook'a tekrar tekrar deploy etmeyi bitirir. **Bir kez** çalıştırırsın,
sabit bir web adresi alırsın; sonra GitHub'a her push otomatik olarak canlıya
yansır — bir daha terminale dokunmana gerek yoktur.

```bash
python -m pip install modal
python -m modal setup
modal deploy modal/remedia_web_v2.py
```

Komut kalıcı bir URL yazdırır (örn. `https://<workspace>--remedia-web-web.modal.run`).
Formda UniProt ID, molekül sayısı, üretici, poz motoru ve **Hız** seçilir.

### Neden tekrar deploy gerekmiyor

Uygulama kodu artık imaja donmuyor; her iş çalıştığında ve her sayfa
yenilemesinde GitHub'daki takip edilen daldan (`main`) çekiliyor:

- `run_job` başlarken `git fetch` + `reset --hard` ile en güncel koda geçer.
- Ana sayfa yenilendiğinde (kısıtlamalı) aynı senkron tetiklenir.
- Sonuçlar, cache ve iş dosyaları repo dizininin **dışında** durduğu için
  güncelleme hiçbir çıktı verisini silmez.
- GitHub'a erişilemezse imaja gömülü kopyaya düşer, uygulama yine açılır.

Başka bir dala sabitlemek istersen ortam değişkeni ver:

```bash
REMEDIA_GIT_BRANCH=stabil modal deploy modal/remedia_web_v2.py
```

Kod güncellemek için tek yapman gereken GitHub'daki dala push etmek; ardından
URL'i yenile.

### Hız seçeneği

- **Hızlı** (varsayılan): tek GNINA taraması (`sadece_fast`). İkinci, yavaş
  ayrıntılı süreci atlar; günlük tarama için belirgin biçimde daha hızlıdır.
- **Dengeli**: iki aşamalı `hızlı → ayrıntılı` doğrulama. Daha yavaş ama nihai
  aday setinde daha sağlam skor verir.

DiffDock/Boltz-2 poz motorlarını seçersen GNINA yalnızca destekleyici rolde
kalır; Hız seçeneği o durumda sadece GNINA kısmını etkiler.

## En kolay yol: form kullanan Modal Notebook

Bu yol bilgisayar terminali ve kod satırı düzenleme gerektirmez.

1. `https://modal.com/notebooks` sayfasını aç.
2. `notebooks/remedia_modal.ipynb` dosyasını yükle.
3. **Compute** panelinden `L4`, `4 CPU`, `8 GiB RAM` seç.
4. Kalıcı dosya istiyorsan `remedia-data` adlı Volume oluştur ve
   `/mnt/remedia-data` yoluna bağla.
5. Notebook'taki tek kod hücresini çalıştır.
6. Açılan formdan reseptörü, UniProt ID'yi, molekül sayısını ve doğruluk profilini seç.
7. **Remedia'yı Başlat** düğmesine bas.

Formda hazır reseptör seçenekleri, özel UniProt ID kutusu, yöntem seçimi,
molekül sayısı, balanced/final doğruluk profili ve gelişmiş docking ayarları
bulunur. Kod satırı değiştirmek gerekmez.

Notebook eksik Python paketlerini, fpocket'i, GNINA'yı ve Remedia kodunu kendi
kendine kurar. Volume bağlıysa kurulum cache'i, pocket cache ve sonuçlar sonraki
oturumlarda korunur.

## Daha hızlı tekrar açılış: özel Modal imajı

Bilgisayar terminalinde:

```bash
git clone https://github.com/mehmetg06/Remedia.git
cd Remedia
python -m pip install modal
python -m modal setup
modal run modal/remedia_modal.py --timeout-minutes 60
```

Komut özel CUDA imajını oluşturur, L4 GPU açar ve token korumalı JupyterLab
bağlantısını terminale yazdırır. Oturum varsayılan olarak 60 dakika sonra
otomatik kapanır.

Başka GPU:

```bash
REMEDIA_MODAL_GPU=L40S modal run modal/remedia_modal.py --timeout-minutes 60
```

Kod güncellendiyse kalıcı Volume'daki kopyayı yenile:

```bash
modal run modal/remedia_modal.py --timeout-minutes 60 --refresh-code
```

## Hosted Modal Notebook'ta özel imajı kullanma

Özel imajı Modal hesabına kaydetmek için:

```bash
modal deploy modal/remedia_modal.py
```

Ardından Modal Notebook'un Compute/Image bölümünde `remedia-modal` uygulamasının
`notebook_image` fonksiyonunu seç ve `remedia-data` Volume'unu bağla.

## Kalıcı yollar

```text
/workspace/Remedia_results/   Özel Jupyter başlatıcısı
/workspace/remedia_cache/     Özel Jupyter başlatıcısı
/mnt/remedia-data/            Hosted Modal Notebook Volume'u
```

## Harcama güvenliği

- Varsayılan GPU `L4`.
- Formun varsayılan doğruluk profili `balanced`.
- Benchmark varsayılan olarak kapalıdır.
- Jupyter başlatıcısı varsayılan 60 dakika, en fazla 240 dakika çalışır.
- Hosted Modal Notebook'ta idle shutdown değerini 10 dakika bırak.
- Modal dashboard'dan Workspace ve Environment Budget belirle.
