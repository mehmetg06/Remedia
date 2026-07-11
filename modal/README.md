# Remedia on Modal

Modal için iki çalışma yolu vardır.

## 1. En kolay: Modal Notebooks

Bu yol bilgisayarda terminal gerektirmez.

1. `https://modal.com/notebooks` sayfasını aç.
2. Yeni notebook oluştur ve `notebooks/remedia_modal.ipynb` dosyasını yükle.
3. **Compute** panelinden GPU olarak `L4` seç.
4. Kalıcı dosya istiyorsan Files panelinden `remedia-data` adlı Volume oluştur ve
   `/mnt/remedia-data` yoluna bağla.
5. **Run all** seç.

Notebook eksik Python paketlerini, fpocket'i, GNINA'yı ve Remedia kodunu kendi
kendine kurar. Volume bağlıysa kurulum cache'i, pocket cache ve sonuçlar sonraki
oturumlarda korunur.

Modal Notebooks varsayılan olarak boşta 10 dakika sonra kernel'i durdurabilir.
Bu ayarı düşük bırakmak gereksiz ücret riskini azaltır.

## 2. Daha hızlı tekrar açılış: özel Modal imajı

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
- Jupyter başlatıcısı varsayılan 60 dakika, en fazla 240 dakika çalışır.
- Hosted Modal Notebook'ta idle shutdown değerini 10 dakika bırak.
- Modal dashboard'dan Workspace ve Environment Budget belirle.
