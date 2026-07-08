# legacy/ — Eski / opsiyonel arayüzler

Bu klasördeki dosyalar Remedia'nın **eski** çalıştırma yollarıdır. **Ana akış
artık bunlar DEĞİL.** Ana akış tek bir Colab notebook'udur:
[`notebooks/remedia_pipeline.ipynb`](../notebooks/remedia_pipeline.ipynb).

Bu dosyalar **silinmedi**; referans olsun ve isteyen eski akışları
kullanabilsin diye burada tutuluyor. Aktif olarak bakımı yapılmıyor ve
kök dizindeki `src/` modülleriyle senkron olmayabilirler.

| Dosya / Klasör | Ne işe yarardı |
|---|---|
| `app.py` | Streamlit tabanlı web arayüzü (eski ana UI). |
| `Snakefile` | Snakemake pipeline tanımı (eski otomasyon akışı). |
| `config.yaml` | Snakemake/Streamlit için merkezi konfigürasyon. |
| `setup.sh` | Codespaces/yerel ortam kurulum betiği. |
| `.devcontainer/` | GitHub Codespaces geliştirme konteyneri ayarları. |
| `.streamlit/` | Streamlit UI teması/ayarları. |
| `dashboard.html` | Eski statik sonuç panosu (dashboard). |
| `notebooks/gnina_colab.ipynb` | Yalnızca **docking** adımını yapan eski Colab notebook'u (Codespaces↔Colab hibrit akış). |
| `notebooks/diffdock_colab.ipynb` | DiffDock ile docking deneyen eski Colab notebook'u. |

## Neden taşındı?

Proje **radikal şekilde basitleştirildi**: Codespaces↔Colab hibrit akışı,
dosya taşıma ve git senkronizasyonu kaldırıldı. Artık her şey tek notebook'ta,
baştan sona Colab'da (GPU) çalışıyor. Ayrıntı için kök dizindeki
[`README.md`](../README.md).
