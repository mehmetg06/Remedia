# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
generative_model.py
HAZIR, ÖNCEDEN EĞİTİLMİŞ bir generative model (REINVENT4) ile tohum molekül
GEREKTİRMEDEN, sıfırdan yeni aday moleküller üreten katman.

ÖNEMLİ SINIRLAMA: Bu modül REINVENT4'ün reseptöre özel EĞİTİMİNİ (fine-tuning /
transfer learning / reinforcement learning) YAPMAZ. Yalnızca halka açık,
önceden eğitilmiş genel "prior" modelinden ("sampling" modu) örnekleme yapar —
o model ilaç-benzeri moleküllerin genel kimyasal dağılımını öğrenmiştir,
herhangi bir spesifik reseptöre yönlendirilmemiştir. Reseptöre uygunluk bu
modülün DIŞINDA, mevcut GNINA docking + ADMET filtreleme adımlarıyla test
edilir.

Diğer üretim yöntemlerinden (molecule_generator.py: random/brics/genetic/
fusion) farkı: tohum molekül gerektirmez, kimyasal kural tabanlı değil, RDKit
ile değil bir sinir ağı (RNN/transformer) ile üretim yapar.

Çıktı formatı `molecule_generator.write_smi` ile TAMAMEN AYNIDIR, böylece
docking → ADMET → sıralama zincirine doğrudan girebilir.

Kullanım (CLI):
    python src/generative_model.py --n 30 --output data/generated_reinvent.smi

Kullanım (Python API):
    from generative_model import generate_with_reinvent
    smiles = generate_with_reinvent(num_molecules=30, output_path="data/generated_reinvent.smi")
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# molecule_generator.py ile aynı çıktı formatını / makul-molekül eleğini
# kullanmak için doğrudan ondan içe aktarıyoruz (format iki yerde AYRI
# tanımlanırsa zamanla birbirinden sapabilir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from molecule_generator import write_smi, is_reasonable, canonical_or_none  # noqa: E402

REINVENT_REPO_URL = "https://github.com/MolecularAI/REINVENT4.git"
# Zenodo'daki resmi prior model deposu (REINVENT4 README'sinde referans verilen
# DOI: 10.5281/zenodo.15641296 -> en güncel sürüm kaydı 15641297).
ZENODO_RECORD_IDS = ["15641297", "15641296"]
ZENODO_PRIOR_FILENAME = "reinvent.prior"


def _default_install_dir() -> Path:
    """REINVENT4'ün klonlanacağı/kurulacağı yer — mevcut çalışma dizininde,
    Hücre 1'in Remedia reposunu kendi cwd'sine klonlamasıyla AYNI mantık."""
    return Path.cwd() / "REINVENT4"


def _prior_path(install_dir: Path) -> Path:
    return install_dir / "priors" / ZENODO_PRIOR_FILENAME


def is_reinvent_ready(install_dir: Path | None = None) -> bool:
    """Kurulumun TAMAMLANMIŞ olup olmadığını kontrol eder: repo var mı, `reinvent`
    komutu çalışıyor mu, prior ağırlığı indirilmiş mi. Notebook'un 'zaten
    kuruluysa atla' mantığı bunu kullanır."""
    install_dir = install_dir or _default_install_dir()
    if not _prior_path(install_dir).exists():
        return False
    return shutil.which("reinvent") is not None or _reinvent_importable()


def _reinvent_importable() -> bool:
    try:
        import reinvent  # noqa: F401
        return True
    except Exception:
        return False


def install_reinvent(
    install_dir: Path | str | None = None,
    log_fn=print,
    skip_if_ready: bool = True,
) -> Path:
    """REINVENT4'ü GitHub'dan klonlar, pip ile kurar ve resmi Zenodo prior
    ağırlığını indirir. TEK SEFERLİK olacak şekilde idempotenttir: her adım
    zaten tamamlanmışsa atlanır (Lightning AI'daki 'kalıcı disk varsa tekrar
    kurma' mantığıyla aynı).

    Reseptöre özel HİÇBİR eğitim/fine-tuning burada YAPILMAZ — yalnızca genel
    amaçlı, önceden eğitilmiş prior indirilir.
    """
    install_dir = Path(install_dir) if install_dir else _default_install_dir()

    if skip_if_ready and is_reinvent_ready(install_dir):
        log_fn(f"• REINVENT4 zaten kurulu ve hazır: {install_dir}")
        return install_dir

    # --- 1) Repoyu klonla -----------------------------------------------
    if not install_dir.is_dir():
        log_fn(f"• REINVENT4 klonlanıyor: {REINVENT_REPO_URL}")
        subprocess.run(
            ["git", "clone", "--depth", "1", REINVENT_REPO_URL, str(install_dir)],
            check=True,
        )
    else:
        log_fn(f"• REINVENT4 reposu zaten var: {install_dir}")

    # --- 2) Python paketi olarak kur -------------------------------------
    if shutil.which("reinvent") is None and not _reinvent_importable():
        log_fn("• REINVENT4 pip ile kuruluyor (bu birkaç dakika sürebilir, ~3-5GB indirme)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(install_dir)],
            check=True,
        )
        # REINVENT4'ün pyproject.toml'unda EKSİK bir bağımlılık: TensorBoard
        # raporlama kodu (reinvent/runmodes/utils/plot.py) scipy import eder
        # ama scipy pyproject'te tanımlı DEĞİL -> onsuz `reinvent` komutu bile
        # çalışmıyor (ImportError). Test sırasında bulundu; elle tamamlıyoruz.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy"], check=True)
    else:
        log_fn("• REINVENT4 paketi zaten kurulu")

    # --- 3) Önceden eğitilmiş prior ağırlığını indir ---------------------
    # NOT: Bu, REINVENT4'ün "de novo" (LibInvent/LinkInvent değil, düz
    # Reinvent) genel priorudur — HERHANGİ bir reseptöre yönlendirilmemiştir.
    prior_path = _prior_path(install_dir)
    if not prior_path.exists():
        log_fn("• Önceden eğitilmiş prior ağırlığı Zenodo'dan indiriliyor...")
        _download_prior(prior_path, log_fn=log_fn)
    else:
        log_fn(f"• Prior ağırlığı zaten indirilmiş: {prior_path}")

    if not prior_path.exists():
        raise RuntimeError(
            f"Prior ağırlığı indirilemedi ({prior_path}). Zenodo'ya erişim "
            "engelliyse (kurumsal ağ/proxy vb.) manuel indirip bu yola koy: "
            f"https://zenodo.org/records/{ZENODO_RECORD_IDS[0]} -> {ZENODO_PRIOR_FILENAME}"
        )

    log_fn(f"✅ REINVENT4 kurulumu tamam: {install_dir}")
    return install_dir


def _download_prior(dest: Path, log_fn=print) -> None:
    """Zenodo REST API üzerinden kayıttaki dosya listesini sorgular ve
    'reinvent.prior' dosyasını indirir (URL'yi elle tahmin etmek yerine
    API'den gerçek indirme linkini alır — Zenodo dosya adları/urlleri sürüm
    aralarında değişebilir)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for record_id in ZENODO_RECORD_IDS:
        try:
            api_url = f"https://zenodo.org/api/records/{record_id}"
            r = requests.get(api_url, timeout=30)
            r.raise_for_status()
            files = r.json().get("files", [])
            match = next((f for f in files if f.get("key") == ZENODO_PRIOR_FILENAME), None)
            if match is None:
                last_err = f"kayıt {record_id} içinde {ZENODO_PRIOR_FILENAME} bulunamadı"
                continue
            download_url = match["links"]["self"]
            log_fn(f"  -> {download_url}")
            with requests.get(download_url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Zenodo'dan prior indirilemedi: {last_err}")


def _build_sampling_toml(prior_path: Path, num_molecules: int, output_csv: Path,
                          device: str, seed: int | None) -> str:
    """REINVENT4'ün 'sampling' run_type'ı için TOML config metni üretir —
    FINE-TUNING/RL YOK, yalnızca var olan prior'dan örnekleme."""
    lines = [
        'run_type = "sampling"',
        f'device = "{device}"',
        "",
        "[parameters]",
        f'model_file = "{prior_path.as_posix()}"',
        f"output_file = '{output_csv.as_posix()}'",
        f"num_smiles = {int(num_molecules)}",
        "unique_molecules = true",
        "randomize_smiles = true",
    ]
    return "\n".join(lines) + "\n"


def generate_with_reinvent(
    num_molecules: int = 30,
    output_path: str | Path = "generated_reinvent.smi",
    install_dir: Path | str | None = None,
    device: str | None = None,
    seed: int | None = None,
    prefix: str = "reinvent",
    log_fn=print,
) -> list[str]:
    """REINVENT4'ün önceden eğitilmiş prior modelinden `num_molecules` adet
    YENİ molekül örnekler — TOHUM MOLEKÜL GEREKTİRMEZ, reseptöre özel eğitim
    YAPILMAZ. Üretilen her SMILES RDKit ile doğrulanır (geçersiz/sanitize
    edilemeyenler elenir). Sonuç `molecule_generator.write_smi` ile AYNI
    formatta bir .smi dosyasına yazılır.

    Returns:
        Geçerli, kanonik SMILES listesi (docking'e girmeye hazır).
    """
    install_dir = install_reinvent(install_dir=install_dir, log_fn=log_fn)
    prior_path = _prior_path(install_dir)

    if device is None:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Örnekleme genelde talep edilenden fazla üretmek gerektirir; çünkü bir
    # kısmı RDKit doğrulamasından/`is_reasonable` eleğinden düşebilir.
    request_n = max(int(num_molecules * 1.3), num_molecules + 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_csv = tmpdir / "sampling.csv"
        toml_path = tmpdir / "sampling.toml"
        log_path = tmpdir / "sampling.log"
        toml_path.write_text(
            _build_sampling_toml(prior_path, request_n, output_csv, device, seed)
        )

        # TensorFlow ve PyTorch'un aynı anda CUDA'ya erişip segfault'a (çökme) neden
        # olmasını önlemek için (özellikle Colab'da), REINVENT4'ü doğrudan değil,
        # aracı bir betikle çalıştırıyoruz. Bu betik önce TensorFlow'un GPU'ya erişimini
        # kapatır (PyTorch çalışmaya devam eder).
        wrapper_path = tmpdir / "run_reinvent_wrapper.py"
        wrapper_path.write_text(
            "import sys\n"
            "sys.modules['tensorflow'] = None\n"
            "import runpy\n"
            "if __name__ == '__main__':\n"
            "    runpy.run_module('reinvent.Reinvent', run_name='__main__')\n"
        )

        cmd = [sys.executable, str(wrapper_path)]
        cmd += ["-l", str(log_path), "-d", device]
        if seed is not None:
            cmd += ["-s", str(int(seed))]
        cmd += [str(toml_path)]

        log_fn(f"• REINVENT4 sampling çalıştırılıyor (device={device}, n={request_n})...")
        
        # Ekstra güvenlik için TF'in tüm belleği almasını engelleyen env değişkeni
        env = os.environ.copy()
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            log_fn(result.stdout[-3000:])
            log_fn(result.stderr[-3000:])
            raise RuntimeError(
                f"REINVENT4 sampling başarısız (exit={result.returncode}). "
                "Ayrıntı için yukarıdaki çıktıya bak."
            )

        if not output_csv.exists():
            raise RuntimeError(f"REINVENT4 çıktı dosyası bulunamadı: {output_csv}")

        raw_smiles = _read_smiles_column(output_csv)

    # --- RDKit ile doğrula + molecule_generator.is_reasonable ile ele ------
    valid: list[str] = []
    seen = set()
    for smi in raw_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = canonical_or_none(mol)
        if canon is None or canon in seen:
            continue
        if not is_reasonable(canon):
            continue
        seen.add(canon)
        valid.append(canon)
        if len(valid) >= num_molecules:
            break

    log_fn(f"✅ {len(valid)}/{len(raw_smiles)} üretilen SMILES RDKit doğrulamasından geçti "
           f"(istenen: {num_molecules}).")

    write_smi(valid, output_path, prefix=prefix)
    return valid


def _read_smiles_column(csv_path: Path) -> list[str]:
    """REINVENT4'ün sampling çıktı CSV'sinden SMILES sütununu okur (Reinvent
    prior için sütunlar: SMILES, SMILES_state, NLL)."""
    import csv

    smiles = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row.get("SMILES")
            if smi:
                smiles.append(smi)
    return smiles


def main():
    parser = argparse.ArgumentParser(
        description="REINVENT4 önceden eğitilmiş prior'dan tohum gerektirmeden molekül üretimi"
    )
    parser.add_argument("--n", type=int, default=30, help="Üretilecek molekül sayısı")
    parser.add_argument("--output", default="data/generated_reinvent.smi", help="Çıktı .smi dosyası")
    parser.add_argument("--install-dir", default=None, help="REINVENT4 kurulum dizini")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (varsayılan: otomatik)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    mols = generate_with_reinvent(
        num_molecules=args.n,
        output_path=args.output,
        install_dir=args.install_dir,
        device=args.device,
        seed=args.seed,
    )
    print(f"[OK] {len(mols)} molekul uretildi (REINVENT4 pretrained prior) -> {args.output}")
    print("     Sonraki adim: bu .smi dosyasini ligand_prep -> docking -> admet -> rank zincirine ver.")


if __name__ == "__main__":
    main()
