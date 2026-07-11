# Remedia on RunPod

## En hızlı yol: Remedia özel imajı

Önerilen template ayarları:

- Image: `ghcr.io/mehmetg06/remedia-runpod:latest`
- HTTP port: `8888`
- Container disk: `20 GB`
- Volume disk: `20 GB`
- Volume mount path: `/workspace`
- Environment variable: `JUPYTER_PASSWORD=remedia`
- GPU: L40S önerilir; RTX 4090/5090 veya A5000 de çalışır.

Pod açıldığında JupyterLab doğrudan
`Remedia/notebooks/remedia_runpod.ipynb` dosyasına yönlenir. Notebook'ta
**Run → Run All Cells** seçmek yeterlidir.

## Resmi PyTorch template ile kurulum

Özel imaj kullanılmayacaksa RunPod'un resmi PyTorch/Jupyter template'iyle Pod
başlat ve Web Terminal'de:

```bash
curl -fsSL https://raw.githubusercontent.com/mehmetg06/Remedia/main/runpod/bootstrap.sh | bash
```

Script repo, Python bağımlılıkları, fpocket ve GNINA'yı kalıcı `/workspace`
volume'una kurar ve notebook bağlantısını yazdırır. Sonraki Pod başlangıçlarında
aynı volume kullanılırsa kurulum tekrar yapılmaz.

## Gerçek tek-tık deploy bağlantısı

RunPod deploy bağlantısı `https://console.runpod.io/deploy?template=TEMPLATE_ID`
biçimindedir. `runpod/template.json` dosyası özel template'in hazır
konfigürasyonudur. Template RunPod hesabında oluşturulduktan sonra oluşan ID
README düğmesine eklenebilir.

> GHCR container paketinin anonim olarak çekilebilmesi için paket görünürlüğü
> GitHub Packages ayarlarından **Public** olmalıdır.
