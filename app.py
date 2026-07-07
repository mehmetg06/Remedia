# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
app.py — Remedia Streamlit arayüzü.

Tek dosyalık, tek komutla açılan web arayüzü:

    streamlit run app.py

TASARIM PRENSİBİ: Basit/gelişmiş diye AYRI mod YOK. Her ekranda ham teknik
değer İLE sade açıklaması YAN YANA, hep birlikte görünür. Teknik kullanıcı ham
sayıları direkt görür; hiç bilmeyen kullanıcı yanındaki cümleden ne olduğunu
anlar. Hiçbir şey gizli/collapsed menüde değildir.

5 adım: (1) Hedef seçimi → (2) Pocket seçimi → (3) Önerilen tohumlar + Üretim
         yöntemi → (4) Çalıştır → (5) Sonuçlar
"""
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import random
import string
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# --- Proje modüllerini import edilebilir kıl ---------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, QED

import molecule_generator as mg  # noqa: E402

# ============================================================================
# SAYFA / TEMA
# ============================================================================
st.set_page_config(page_title="Remedia — Molekül Üretici", page_icon="🧬", layout="wide")

ACCENT = "#5EEAD4"
st.markdown(
    f"""
    <style>
      .plain {{ color:#7E8C9A; font-size:0.85rem; line-height:1.45; }}
      .tech  {{ color:{ACCENT}; font-family:'IBM Plex Mono',monospace; font-weight:600; }}
      .card  {{ background:#10161D; border:1px solid rgba(255,255,255,0.07);
                border-radius:12px; padding:16px 18px; margin-bottom:12px; }}
      .seed-card {{ background:#0D1B2A; border:1px solid rgba(94,234,212,0.18);
                    border-radius:10px; padding:14px 16px; margin-bottom:10px; }}
      .run-tag {{ background:#0F2B2B; color:{ACCENT}; border:1px solid rgba(94,234,212,0.3);
                  border-radius:8px; padding:8px 16px; font-family:monospace; font-size:0.88rem;
                  display:inline-block; margin-bottom:12px; }}
      .status-ok  {{ color:#34D399; font-weight:700; }}
      .status-err {{ color:#F87171; font-weight:700; }}
      .eyebrow {{ color:{ACCENT}; font-family:monospace; letter-spacing:0.15em;
                  font-size:0.72rem; text-transform:uppercase; }}
      h1,h2,h3 {{ font-family:'Space Grotesk',sans-serif; }}
      .stProgress > div > div > div > div {{ background:{ACCENT}; }}
      .log-box {{ background:#070E18; border:1px solid #1E293B; border-radius:8px;
                  padding:12px; font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                  color:#94A3B8; max-height:300px; overflow-y:auto; white-space:pre-wrap; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def dual(label, value, plain, unit=""):
    """Bir alanın [teknik etiket + ham değer] ile [sade açıklama]'sını yan yana gösterir."""
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(
            f"<div><span style='color:#7E8C9A;font-size:0.8rem'>{label}</span><br>"
            f"<span class='tech' style='font-size:1.15rem'>{value}{unit}</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(f"<div class='plain'>{plain}</div>", unsafe_allow_html=True)


# ============================================================================
# BÖLÜM B — ARAÇ KURULUM KONTROLÜ (uygulama başlangıcında)
# ============================================================================

def _check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _check_python_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@st.cache_data(ttl=60)
def check_tools() -> dict[str, bool]:
    # Docking motoru GNINA'ya (Colab, GPU) taşındı — yerel Vina artık gerekmez.
    return {
        "snakemake": _check_tool("snakemake"),
        "fpocket": _check_tool("fpocket"),
        "obabel": _check_tool("obabel"),
    }


# ============================================================================
# BÖLÜM C — run_id yardımcıları
# ============================================================================

def generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"run_{ts}_{suffix}"


def get_all_run_ids() -> list[str]:
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return []
    runs = sorted(
        [d.name for d in results_dir.iterdir()
         if d.is_dir() and d.name.startswith("run_")],
        reverse=True,
    )
    return runs


def get_latest_run_id() -> str | None:
    latest_txt = ROOT / "results" / "latest_run.txt"
    if latest_txt.exists():
        rid = latest_txt.read_text().strip()
        if rid:
            return rid
    # Fallback: klasörden bul
    runs = get_all_run_ids()
    return runs[0] if runs else None


def run_dir_path(run_id: str) -> Path:
    return ROOT / "results" / run_id


# ============================================================================
# FÜZYON — ayrı süreç (subprocess) + ilerleme dosyası (Sorun 1)
# Füzyon/GA, Streamlit'in ana script akışından TAMAMEN AYRILIR: ayrı bir
# subprocess olarak (molecule_generator.py --method fusion) çalışır. Ana thread
# beklemez; ilerlemeyi bir JSON dosyasından okuyup gösterir. Böylece log_fn
# kaynaklı rerun'lar süreci baştan başlatamaz.
# ============================================================================
# Füzyon ara dosyaları (results/ altındaki "run_" ön ekiyle ÇAKIŞMAZ, yani
# geçmiş pipeline çalıştırmaları listesini kirletmez).
FUSION_WORK_ROOT = ROOT / "results" / "_fusion_runs"


def _pid_alive(pid) -> bool:
    """Verilen PID'li süreç hâlâ yaşıyor mu? (Çift-başlatma koruması için.)
    Zombie (savunmasızca çöken ve reap edilmemiş) süreçleri ÖLÜ sayar; aksi
    halde çöken bir alt-süreç monitörü sonsuz döngüde bırakabilir."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Linux: /proc üzerinden zombie durumunu tespit et.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        state = stat.rsplit(")", 1)[1].split()[0]
        if state == "Z":
            return False
    except Exception:
        pass
    return True


def _read_progress(path) -> dict | None:
    """İlerleme JSON'unu güvenle okur (yazım anına denk gelirse None döner)."""
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except Exception:
        return None


def launch_fusion(seeds: list[str], docking_opts: dict | None, ss) -> bool:
    """Füzyonu AYRI bir subprocess olarak başlatır. Zaten canlı bir süreç varsa
    YENİ süreç KESİNLİKLE başlatmaz (Sorun 1). Başlattıysa True döner."""
    if ss.get("fusion_active") and _pid_alive(ss.get("fusion_process_pid")):
        return False  # zaten çalışıyor

    fid = (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_"
           + "".join(random.choices(string.ascii_lowercase + string.digits, k=4)))
    work = FUSION_WORK_ROOT / fid
    work.mkdir(parents=True, exist_ok=True)

    seeds_file = work / "seeds.smi"
    seeds_file.write_text("\n".join(seeds) + "\n")
    progress_file = work / "progress.json"
    output_file = work / "generated.smi"

    cmd = [
        sys.executable, str(SRC / "molecule_generator.py"),
        "--method", "fusion",
        "--seeds-file", str(seeds_file),
        "--output", str(output_file),
        "--progress-file", str(progress_file),
    ]
    if docking_opts:
        cmd += [
            "--receptor", str(docking_opts["receptor"]),
            "--center", *[str(x) for x in docking_opts["center"]],
            "--size", *[str(x) for x in docking_opts["box_size"]],
            "--exhaustiveness", str(docking_opts.get("exhaustiveness", 8)),
            "--workdir", str(work / "ga_work"),
        ]

    # start_new_session: süreç kendi oturumunda çalışsın (Streamlit yeniden
    # çalışsa/bağlantı kopsa bile devam etsin). Çıktıyı yutuyoruz; ilerleme
    # zaten progress.json'a yazılıyor.
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    ss["fusion_process_pid"] = proc.pid
    ss["fusion_progress_file"] = str(progress_file)
    ss["fusion_output_file"] = str(output_file)
    ss["fusion_active"] = True
    return True


def render_fusion_monitor(ss):
    """İlerleme dosyasını okuyup KOMPAKT bir st.progress() çubuğu + 'X/Y molekül,
    tahmini kalan Z dk' gösterir. Süreç çalışırken kısa aralıklarla st.rerun()
    yapar — ama SADECE dosyayı okumak için; subprocess YENİDEN BAŞLAMAZ."""
    prog = _read_progress(ss.get("fusion_progress_file"))
    alive = _pid_alive(ss.get("fusion_process_pid"))
    status = (prog or {}).get("status", "running")

    done = int((prog or {}).get("done", 0) or 0)
    total = int((prog or {}).get("total", 0) or 0)
    stage = (prog or {}).get("stage", "Başlıyor")
    best = (prog or {}).get("best")
    elapsed = float((prog or {}).get("elapsed", 0.0) or 0.0)

    # ── Tamamlandı: sonuçları session_state'e al (senkron yolla aynı format) ──
    if status == "done" and prog is not None:
        res = []
        for item in prog.get("results", []):
            try:
                smi, aff = item[0], float(item[1])
            except Exception:
                continue
            if smi and Chem.MolFromSmiles(smi):
                res.append((smi, aff))
        res.sort(key=lambda x: x[1])
        ss["results"] = res
        ss["mode"] = prog.get("mode", "qed_fallback")
        ss["fusion_active"] = False
        st.progress(1.0)
        st.success(f"✅ Füzyon tamamlandı — {len(res)} aday üretildi ve skorlandı.")
        # Otomatik doğrulama özeti/hatası — sessizce yutma, kullanıcı görsün.
        _val_msg = prog.get("validation")
        if _val_msg:
            if str(_val_msg).startswith("✅"):
                st.success(_val_msg)
            else:
                st.warning(_val_msg)
        return

    if status == "error":
        ss["fusion_active"] = False
        st.error(f"🔴 Füzyon hatası: {(prog or {}).get('error', 'bilinmiyor')}")
        return

    if not alive:
        # Süreç bitti ama 'done' yazmadı → beklenmedik sonlanma.
        ss["fusion_active"] = False
        st.error("🔴 Füzyon süreci beklenmedik şekilde sonlandı. Lütfen tekrar dene.")
        return

    # ── Hâlâ çalışıyor: kompakt ilerleme çubuğu ──────────────────────────────
    frac = max(0.0, min(1.0, (done / total) if total > 0 else 0.0))
    st.progress(frac)

    if done > 0 and total > done and elapsed > 0:
        remaining = elapsed / done * (total - done)
        eta = f"~{remaining / 60:.1f} dk" if remaining >= 60 else f"~{remaining:.0f} sn"
    else:
        eta = "hesaplanıyor…"
    best_txt = (f" · en iyi <span class='tech'>{best:.2f} kcal/mol</span>"
                if isinstance(best, (int, float)) else "")
    st.markdown(
        f"<div class='plain'>⚡ <b>{stage}</b> · <b>{done}/{total}</b> molekül · "
        f"tahmini kalan: <b>{eta}</b>{best_txt}</div>",
        unsafe_allow_html=True,
    )

    # Kısa bekle ve yalnızca dosyayı yeniden okumak için rerun et.
    time.sleep(1.0)
    st.rerun()


# ============================================================================
# YARDIMCI: PDB analiz
# ============================================================================
def analyze_pdb(pdb_path: Path) -> dict:
    text = pdb_path.read_text()
    n_atoms = 0
    plddt_sum = 0.0
    plddt_count = 0
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            n_atoms += 1
            try:
                b = float(line[60:66])
                plddt_sum += b
                plddt_count += 1
            except ValueError:
                pass
    avg_plddt = plddt_sum / plddt_count if plddt_count else 0.0
    return {
        "size_kb": pdb_path.stat().st_size / 1024,
        "n_atoms": n_atoms,
        "avg_plddt": avg_plddt,
    }


def plddt_comment(v: float) -> str:
    if v >= 90:
        return f"pLDDT {v:.1f} çok yüksek — modelin bu yapıyı çok güvenilir tahmin ettiği anlamına gelir."
    if v >= 70:
        return f"pLDDT {v:.1f} iyi sayılır — yapının çoğu güvenilir, docking için uygun."
    return f"pLDDT {v:.1f} düşük — yapı belirsiz olabilir, sonuçları temkinli yorumla."


# ============================================================================
# YARDIMCI: molekül özellikleri
# ============================================================================
def mol_properties(smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    return {
        "MW": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "QED": round(QED.qed(mol), 3),
    }


def interpret(smi: str, affinity: float, props: dict) -> str:
    bind = ("proteine güçlü bağlanıyor" if affinity <= -8
            else "proteine orta düzeyde bağlanıyor" if affinity <= -6
            else "proteine zayıf bağlanıyor")
    mw_ok = props.get("MW", 999) <= 500
    logp_ok = props.get("LogP", 99) <= 5
    absorb = ("vücutta emilim açısından uygun görünüyor" if mw_ok and logp_ok
              else "emilim açısından bazı riskler taşıyor (molekül ağırlığı/yağda çözünürlük sınırda)")
    return f"Bu molekül {bind} ve {absorb}."


def draw_mol(smi: str, size=(320, 240)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def mol_to_png_bytes(smi: str, size=(200, 160)) -> bytes | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# BÖLÜM B — Pipeline çalıştırma yardımcıları
# ============================================================================

def _validate_docking_csv(csv_path) -> tuple[bool, str]:
    """Colab'dan yüklenen docking_scores.csv'yi src/docking.py ile doğrular
    (ligand,affinity_kcal_mol formatı). (ok, mesaj) döndürür."""
    try:
        from docking import validate_csv_format  # src/ sys.path'te
        return validate_csv_format(str(csv_path))
    except Exception as e:  # noqa: BLE001
        return False, f"🔴 docking_scores.csv doğrulanamadı: {e}"


def _interpret_snakemake_error(stderr: str, stdout: str) -> str:
    """Ham Snakemake hata çıktısını sade Türkçe'ye çevirir."""
    combined = (stderr + stdout).lower()
    if "docking bekleniyor" in combined or ("docking_scores.csv" in combined and "bulun" in combined):
        return ("🔴 Docking sonucu bulunamadı. Önce Colab'da GNINA'yı çalıştırıp inen "
                "`docking_scores.csv`'yi `results/<run_id>/` klasörüne yükle, sonra "
                "\"Docking Tamamlandı, Devam Et\" butonuna bas.")
    if "obabel" in combined and "not found" in combined:
        return "🔴 Ligand hazırlama adımında hata: Open Babel (obabel) kurulu değil. Terminalde şunu çalıştır: `bash setup.sh`"
    if "fpocket" in combined and "not found" in combined:
        return "🔴 Pocket tespitinde hata: fpocket kurulu değil. Terminalde şunu çalıştır: `bash setup.sh`"
    if "modulenotfounderror" in combined or "importerror" in combined:
        return "🔴 Python modülü eksik. Terminalde şunu çalıştır: `bash setup.sh`"
    if "missinginputexception" in combined or "missing input" in combined:
        return "🔴 Girdi dosyası eksik. Adım 1'de yapıyı indirip Adım 4'te molekülleri kaydettiğinden emin ol."
    if "workflowerror" in combined:
        return "🔴 Snakemake iş akışı hatası. Aşağıdaki detaya bak."
    return "🔴 Pipeline hatası oluştu. Aşağıdaki detaya bak."


def _run_snakemake_live(cmd: list[str], log_placeholder, status_placeholder) -> tuple[int, str, str]:
    """
    Snakemake'i arka planda çalıştırır ve log kutusunu canlı günceller.
    Returns (returncode, stdout, stderr)
    """
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    lock = threading.Lock()

    def _stream(pipe, collector):
        for raw in pipe:
            line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
            with lock:
                collector.append(line.rstrip())

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(ROOT),
        text=True,
        bufsize=1,
    )
    t_out = threading.Thread(target=_stream, args=(proc.stdout, stdout_lines), daemon=True)
    t_err = threading.Thread(target=_stream, args=(proc.stderr, stderr_lines), daemon=True)
    t_out.start()
    t_err.start()

    # Streamlit'te canlı güncelleme (polling)
    import time
    while proc.poll() is None:
        time.sleep(0.8)
        with lock:
            all_lines = stdout_lines + stderr_lines
        displayed = "\n".join(all_lines[-40:]) if all_lines else "Bekleniyor..."
        log_placeholder.markdown(
            f"<div class='log-box'>{displayed}</div>",
            unsafe_allow_html=True,
        )
        # Hangi kural çalışıyor?
        for line in reversed(all_lines):
            if "rule " in line.lower() or "Running" in line:
                status_placeholder.markdown(
                    f"<div class='plain'>⚙️ <b>{line.strip()}</b></div>",
                    unsafe_allow_html=True,
                )
                break

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    with lock:
        all_lines = stdout_lines + stderr_lines
    log_placeholder.markdown(
        f"<div class='log-box'>{'<br>'.join(all_lines[-60:])}</div>",
        unsafe_allow_html=True,
    )

    return proc.returncode, "\n".join(stdout_lines), "\n".join(stderr_lines)


# ============================================================================
# BAŞLIK
# ============================================================================
st.markdown("<div class='eyebrow'>REMEDIA · RESEPTÖR ODAKLI İLAÇ KEŞFİ</div>", unsafe_allow_html=True)
st.title("🧬 Yeni Molekül Üretici")
st.markdown(
    "<div class='plain'>Var olan molekülleri test etmenin ötesinde, <b>sıfırdan yeni "
    "aday moleküller üretir</b> — model eğitmeden, yalnızca kimyasal kurallarla. "
    "Her ekranda ham teknik veriyle birlikte sade açıklamasını görürsün.</div>",
    unsafe_allow_html=True,
)

# ── Araç Durumu ──────────────────────────────────────────────────────────────
with st.expander("🔧 Araç Kurulum Durumu", expanded=False):
    tools = check_tools()
    cols = st.columns(len(tools))
    labels = {
        "snakemake": "Snakemake",
        "fpocket": "fpocket",
        "obabel": "Open Babel",
    }
    for (k, v), col in zip(tools.items(), cols):
        icon = "✅" if v else "❌"
        css = "status-ok" if v else "status-err"
        col.markdown(
            f"<span class='{css}'>{icon} {labels[k]}</span>"
            + ("<br><span style='font-size:0.72rem;color:#7E8C9A'>Terminalde: <code>bash setup.sh</code></span>" if not v else ""),
            unsafe_allow_html=True,
        )
    if not all(tools.values()):
        st.markdown(
            "<div class='plain' style='margin-top:6px'>Eksik araçları tek seferde kurmak için "
            "terminalde <code>bash setup.sh</code> çalıştır (idempotent'tir, defalarca çalıştırılabilir).</div>",
            unsafe_allow_html=True,
        )

st.divider()

# Oturum durumu
ss = st.session_state
ss.setdefault("pdb_info", None)
ss.setdefault("uniprot", "P30405")
ss.setdefault("pocket", None)
ss.setdefault("results", None)
ss.setdefault("validated_data", {})
ss.setdefault("known_ligands", None)          # Bölüm A
ss.setdefault("known_ligands_msg", "")        # Bölüm A
ss.setdefault("pipeline_running", False)      # Bölüm B
ss.setdefault("current_run_id", None)         # Bölüm C
ss.setdefault("gnina_run_id", None)           # GNINA/Colab docking için hazırlanan run
ss.setdefault("pipeline_done", False)         # Bölüm B
ss.setdefault("prepared_receptor", None)      # "Reseptörü Hazırla" butonu
ss.setdefault("fusion_active", False)         # Füzyon ayrı süreçte çalışıyor mu
ss.setdefault("fusion_process_pid", None)     # Füzyon subprocess PID'i (çift-başlatma koruması)
ss.setdefault("fusion_progress_file", None)   # İlerleme JSON dosyasının yolu
ss.setdefault("fusion_output_file", None)     # Üretilen .smi çıktısının yolu

try:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
except Exception:
    cfg = {}


def _refresh_known_ligands(uniprot_id: str) -> None:
    """Bölüm A: hedef için ChEMBL'den bilinen inhibitörleri çeker ve session_state'e yazar."""
    if not ss["pdb_info"]:
        return
    with st.spinner(f"🔍 {uniprot_id} için bilinen inhibitörler ChEMBL'de aranıyor..."):
        try:
            from known_ligands import fetch_known_ligands
            ligands, msg = fetch_known_ligands(uniprot_id, max_results=5)
            ss["known_ligands"] = ligands
            ss["known_ligands_msg"] = msg
        except Exception as exc:
            ss["known_ligands"] = []
            ss["known_ligands_msg"] = f"⚠️ Bilinen ligand araması başarısız: {exc}"

# ============================================================================
# ADIM 1 — HEDEF SEÇİMİ
# ============================================================================
st.header("Adım 1 · Hedef Protein")
c1, c2 = st.columns([1, 1.4])
with c1:
    uniprot = st.text_input("UniProt ID", value=ss["uniprot"], key="uniprot_in")
with c2:
    st.markdown(
        "<div class='plain'>Bu, aradığımız proteinin resmî kimlik numarası. "
        "<b>P30405 = CypD</b> — kalp/beyin hasarıyla (iskemi-reperfüzyon) ilişkili "
        "hedefimiz. Farklı bir hedef için başka bir UniProt ID gir.</div>",
        unsafe_allow_html=True,
    )

if st.button("⬇️ Yapıyı İndir (AlphaFold DB)", type="primary"):
    ss["uniprot"] = uniprot
    # UniProt değişti mi? Bilinen ligandları sıfırla
    ss["known_ligands"] = None
    ss["known_ligands_msg"] = ""
    try:
        with st.spinner(f"{uniprot} yapısı AlphaFold DB'den indiriliyor..."):
            import fetch_structure
            pdb_path = fetch_structure.fetch_alphafold(uniprot)
            ss["pdb_info"] = {"path": str(pdb_path), **analyze_pdb(pdb_path)}
        st.success(f"İndirildi: {pdb_path}")
    except Exception as e:
        st.error(f"İndirme başarısız: {e}")

    _refresh_known_ligands(uniprot)

st.markdown(
    "<div class='plain' style='margin-top:-4px'>Ya da indirme → cep tespiti (fpocket) → "
    "PDBQT dönüşümü (obabel) → en iyi cebi otomatik seçme adımlarının tamamını tek "
    "seferde çalıştır:</div>",
    unsafe_allow_html=True,
)

if st.button("🧪 Reseptörü Hazırla", type="primary"):
    ss["uniprot"] = uniprot
    ss["known_ligands"] = None
    ss["known_ligands_msg"] = ""
    ss["prepared_receptor"] = None

    try:
        with st.spinner(f"1/4 · {uniprot} yapısı AlphaFold DB'den indiriliyor..."):
            import fetch_structure
            pdb_path = fetch_structure.fetch_alphafold(uniprot)
            ss["pdb_info"] = {"path": str(pdb_path), **analyze_pdb(pdb_path)}

        with st.spinner("2/4 · fpocket ile bağlanma cepleri taranıyor..."):
            import pocket_detection
            best = pocket_detection.best_druggable_pocket(pdb_path)

        pdbqt_path = ROOT / "data" / f"{uniprot}_alphafold.pdbqt"
        with st.spinner("3/4 · obabel ile PDBQT dönüşümü yapılıyor..."):
            conv = subprocess.run(
                ["obabel", str(pdb_path), "-O", str(pdbqt_path), "-xr"],
                capture_output=True, text=True,
            )
            if conv.returncode != 0 or not pdbqt_path.exists():
                raise RuntimeError(f"obabel dönüşümü başarısız: {conv.stderr.strip()}")

        with st.spinner("4/4 · en yüksek Druggability Score'a sahip cep seçiliyor..."):
            box = cfg.get("box_size", [20.0, 20.0, 20.0])
            ss["prepared_receptor"] = {
                "uniprot": uniprot,
                "pdb_path": str(pdb_path),
                "pdbqt_path": str(pdbqt_path),
                "pocket_number": best["pocket_number"],
                "druggability": best["druggability"],
                "score": best["score"],
                "volume": best["volume"],
                "center": best["center"],
            }
            ss["pocket"] = {
                "name": f"Pocket {best['pocket_number']}",
                "center": list(best["center"]),
                "box": box,
            }

        cx, cy, cz = best["center"]
        st.success(
            f"✅ Reseptör hazır: **{uniprot}**, **Pocket {best['pocket_number']}**, "
            f"merkez ({cx:.2f}, {cy:.2f}, {cz:.2f}) — `--receptor` ve `--center` alanları "
            f"otomatik dolduruldu."
        )
    except Exception as e:
        st.error(f"🔴 Reseptör hazırlama başarısız: {e}")

    _refresh_known_ligands(uniprot)

if ss.get("prepared_receptor"):
    pr = ss["prepared_receptor"]
    cx, cy, cz = pr["center"]
    st.markdown(
        f"<div class='run-tag'>🧪 Reseptör hazır: <b>{pr['uniprot']}</b> · "
        f"Pocket <b>{pr['pocket_number']}</b> (Druggability {pr['druggability']:.3f}) · "
        f"merkez <b>({cx:.2f}, {cy:.2f}, {cz:.2f})</b></div>",
        unsafe_allow_html=True,
    )

if ss["pdb_info"]:
    info = ss["pdb_info"]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    dual("Dosya boyutu", f"{info['size_kb']:.0f}", "İndirilen yapı dosyasının ham boyutu (kilobayt).", " KB")
    dual("Atom sayısı", f"{info['n_atoms']}", "Proteinin toplam atom sayısı — büyük protein = daha çok atom.")
    dual("Ortalama pLDDT", f"{info['avg_plddt']:.1f}", plddt_comment(info["avg_plddt"]))
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ============================================================================
# ADIM 2 — POCKET SEÇİMİ
# ============================================================================
st.header("Adım 2 · Bağlanma Cebi (Pocket)")
st.markdown(
    "<div class='plain'>Bir ilaç molekülünün proteine tutunduğu girinti/oyuğa "
    "<b>cep (pocket)</b> denir. Aşağıdaki tüm teknik metrikler açıkça gösterilir; "
    "her başlığın (?) ikonuna gelince ne anlama geldiğini görürsün.</div>",
    unsafe_allow_html=True,
)

POCKET_HELP = {
    "Pocket": "fpocket'in bulduğu cebin sıra numarası.",
    "Score": "fpocket'in cebe verdiği genel skor — yüksek = daha belirgin cep.",
    "Druggability": "Cebin 'ilaçlanabilirlik' skoru (0–1). 1'e yakın = küçük molekülle hedeflenmeye uygun.",
    "Volume (Å³)": "Cebin hacmi. Çok küçük cepe molekül sığmaz, çok büyük cep seçici değildir.",
    "Apolar SASA": "Cebin suyu iten (yağ sever) yüzey oranı — ilaç bağlanmasında önemli.",
    "Alpha spheres": "fpocket'in cebi tarif eden geometrik küre sayısı — cebin büyüklük göstergesi.",
    "Flexibility": "Cebin esnekliği (0–1). Yüksek = hareketli, bağlanmayı zorlaştırabilir.",
}

pk = (cfg.get("dashboard", {}) or {}).get("pocket", {})
box = cfg.get("box_size", [20.0, 20.0, 20.0])

import pandas as pd
pr = ss.get("prepared_receptor")
if pr:
    center = list(pr["center"])
    pocket_rows = [{
        "Pocket": f"Pocket {pr['pocket_number']}",
        "Score": pr.get("score", 0.0),
        "Druggability": pr.get("druggability", 0.0),
        "Volume (Å³)": pr.get("volume", 0.0),
        "Apolar SASA": pk.get("apolar_sasa", "—"),
        "Alpha spheres": pk.get("alpha_spheres", "—"),
        "Flexibility": pk.get("flexibility", "—"),
    }]
else:
    center = cfg.get("pocket_center", [5.00, -1.02, -15.56])
    pocket_rows = [{
        "Pocket": (cfg.get("dashboard", {}) or {}).get("pocket_name", "Pocket 9"),
        "Score": pk.get("druggability", 0.168) * 100 if pk else 16.8,
        "Druggability": pk.get("druggability", 0.168),
        "Volume (Å³)": pk.get("volume", 384.8),
        "Apolar SASA": pk.get("apolar_sasa", 0.871),
        "Alpha spheres": pk.get("alpha_spheres", 31),
        "Flexibility": pk.get("flexibility", 0.993),
    }]
pocket_df = pd.DataFrame(pocket_rows)

st.dataframe(
    pocket_df,
    width="stretch",
    hide_index=True,
    column_config={
        col: st.column_config.Column(help=POCKET_HELP.get(col, ""))
        for col in pocket_df.columns
    },
)

sel = st.radio(
    "Docking için cebi seç:",
    options=list(pocket_df["Pocket"]),
    horizontal=True,
)
ss["pocket"] = {"name": sel, "center": center, "box": box}
dual("Cep merkezi (x, y, z)", f"{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}",
     "Docking kutusunun protein üzerindeki merkez koordinatları (Ångström).")
dual("Kutu boyutu (Å)", f"{box[0]:.0f} × {box[1]:.0f} × {box[2]:.0f}",
     "Moleküllerin yerleştirilip denendiği kübik arama hacmi.")

st.divider()

# ============================================================================
# ADIM 3 — ÖNERİLEN TOHUMLAR (BÖLÜM A) + ÜRETİM YÖNTEMİ
# ============================================================================
st.header("Adım 3 · Tohum Moleküller & Üretim Yöntemi")
st.markdown(
    "<div class='plain'>Yeni moleküllerin nasıl türetileceğini seç. Her yöntemin "
    "parametreleri varsayılan değerleriyle <b>doğrudan görünür</b> — istersen değiştir, "
    "istemezsen olduğu gibi bırak.</div>",
    unsafe_allow_html=True,
)

# ── BÖLÜM A: Önerilen Tohumlar ───────────────────────────────────────────────
known_ligands: list[dict] = ss.get("known_ligands") or []
known_msg: str = ss.get("known_ligands_msg", "")

# Varsayılanlar: aşağıdaki dalların herhangi biri atlanırsa bile tanımlı kalsın
# (örn. yapı henüz indirilmediyse Tohum kutusu yine de çalışmalı).
FALLBACK_SEEDS = "CC(=O)Oc1ccccc1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
default_seeds = FALLBACK_SEEDS
selected_smiles: list[str] = []

# Eğer hiç çağrılmadıysa butona bas çıktısı yoktur; küçük not göster
if ss["pdb_info"] is None:
    st.markdown(
        "<div class='plain'>ℹ️ Adım 1'de yapıyı indirdikten sonra bu hedef için "
        "bilinen inhibitörler otomatik olarak önerilecek.</div>",
        unsafe_allow_html=True,
    )
elif known_ligands:
    st.markdown("### 🔬 Önerilen Tohumlar")
    if known_msg:
        st.markdown(f"<div class='plain'>{known_msg}</div>", unsafe_allow_html=True)

    selected_smiles: list[str] = []
    seed_cols = st.columns(min(len(known_ligands), 3))

    for idx, lig in enumerate(known_ligands):
        col = seed_cols[idx % len(seed_cols)]
        with col:
            st.markdown("<div class='seed-card'>", unsafe_allow_html=True)
            # 2D çizim
            png = mol_to_png_bytes(lig["smiles"], size=(200, 150))
            if png:
                st.image(png, width="stretch")
            # Bilgi
            st.markdown(
                f"<b style='font-size:0.9rem'>{lig['name']}</b><br>"
                f"<span style='color:#5EEAD4;font-size:0.78rem'>{lig['source']}</span><br>"
                f"<span style='color:#94A3B8;font-size:0.76rem'>{lig['activity']}</span><br>"
                f"<span style='color:#64748B;font-size:0.7rem;word-break:break-all'>"
                f"{lig['smiles'][:50]}{'…' if len(lig['smiles']) > 50 else ''}</span>",
                unsafe_allow_html=True,
            )
            checked = st.checkbox(
                f"Tohum olarak kullan",
                value=True,
                key=f"seed_check_{idx}",
                help=f"Bu molekül {lig['source']}'de bu hedefe karşı {lig['activity']} ile kayıtlı.",
            )
            if checked and lig["smiles"]:
                selected_smiles.append(lig["smiles"])
            st.markdown("</div>", unsafe_allow_html=True)

    # Şeffaflık notu
    st.markdown(
        "<div class='plain' style='font-size:0.78rem;margin-top:4px'>"
        "ℹ️ Bu moleküller veritabanı kayıtlarından otomatik getirildi. "
        "Seçtiklerini aşağıdaki kutuya aktardık — istersen elle de düzenleyebilirsin.</div>",
        unsafe_allow_html=True,
    )

    default_seeds = "\n".join(selected_smiles) if selected_smiles else FALLBACK_SEEDS
else:
    # Hiç bulunamadı
    if known_msg:
        st.info(known_msg)
    default_seeds = FALLBACK_SEEDS
    selected_smiles = []

# ── Tohum kutusu (elle düzenlenebilir) ────────────────────────────────────────
# Seçimlerdeki değişiklikler checkbox callback'leriyle default değeri etkiler.
# Kullanıcı text_area'yı her zaman elle düzenleyebilir.
seeds_text = st.text_area(
    "Tohum molekül(ler) — her satıra bir SMILES",
    value=default_seeds,
    height=120,
    help="Üretim bu 'başlangıç' moleküllerinden yola çıkar. Yukarıdaki önerilerden seçim yapılır ya da elle yazılır.",
    key="seeds_textarea",
)
seeds = [s.strip().split()[0] for s in seeds_text.splitlines() if s.strip() and not s.startswith("#")]

method = st.radio(
    "Yöntem",
    ["fusion", "random", "brics", "genetic", "pretrained"],
    format_func=lambda m: {
        "fusion": "⚡ Füzyon (Önerilen) — tüm yöntemleri akıllıca birleştirir",
        "random": "🎲 Random Mutation",
        "brics": "🧩 BRICS Fragment Recombination",
        "genetic": "🧬 Genetic Algorithm",
        "pretrained": "🤖 Pretrained Model (opsiyonel)",
    }[m],
)

col_a, col_b = st.columns([1, 1.4])
params = {}
if method == "fusion":
    with col_a:
        st.markdown("<div class='plain'>Parametreler otomatik ayarlanır.</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<div class='plain'><b>Füzyon (Önerilen)</b> — Önce çok sayıda çeşitli molekül üretir, ucuz filtrelerle en umut vericileri seçer, sonra sadece onları pahalı docking'le derinlemesine optimize eder.</div>",
                    unsafe_allow_html=True)
elif method == "random":
    with col_a:
        params["n"] = st.number_input("Üretilecek molekül sayısı (n)", 5, 500, 50)
    with col_b:
        st.markdown("<div class='plain'><b>Random Mutation</b> — Tohum moleküldeki "
                    "atomları rastgele değiştirir (C↔N↔O↔S) ve küçük gruplar "
                    "(metil, hidroksil, halojen) ekler/çıkarır. Geçersiz moleküller "
                    "otomatik elenir.<br><i>n: kaç yeni molekül üretilsin.</i></div>",
                    unsafe_allow_html=True)
elif method == "brics":
    with col_a:
        params["n"] = st.number_input("Üretilecek molekül sayısı (n)", 5, 500, 50)
    with col_b:
        st.markdown("<div class='plain'><b>BRICS Fragment Recombination</b> — Var olan "
                    "moleküllerin parçalarını LEGO gibi söküp yeniden birleştirerek "
                    "yeni moleküller oluşturur. Birden fazla tohum verirsen çeşitlilik "
                    "artar.<br><i>n: kaç yeni molekül üretilsin.</i></div>",
                    unsafe_allow_html=True)
elif method == "genetic":
    with col_a:
        params["generations"] = st.number_input("Nesil sayısı", 1, 100, 10,
                                                 help="Kaç kuşak boyunca iyileştirme yapılsın.")
        params["population"] = st.number_input("Popülasyon boyutu", 5, 200, 30,
                                               help="Her nesilde kaç molekül yarışsın.")
        params["mutation_rate"] = st.slider("Mutasyon oranı", 0.0, 1.0, 0.30,
                                            help="Yeni bireylerin ne kadarı mutasyonla üretilsin.")
    with col_b:
        st.markdown("<div class='plain'><b>Genetic Algorithm</b> — Doğal seçilimi taklit "
                    "eder: her nesilde molekülleri docking ile skorlar, en iyi %20'yi "
                    "tutar, gerisini çaprazlama + mutasyonla yeniler. Nesiller ilerledikçe "
                    "skorlar iyileşir.<br>"
                    "<i>Nesil sayısı: kaç tur iyileştirme. Popülasyon: her turdaki molekül "
                    "sayısı. Mutasyon oranı: rastgele değişim payı.</i></div>",
                    unsafe_allow_html=True)
else:  # pretrained
    with col_a:
        params["n"] = st.number_input("İstenen örnek sayısı (n)", 5, 500, 50)
    with col_b:
        st.markdown("<div class='plain'><b>Pretrained Model</b> — REINVENT gibi HAZIR bir "
                    "üretken model kurduysan onu çağırır. Bu <b>opsiyoneldir</b>; kurulu "
                    "değilse sistem diğer üç yöntemle tam çalışır.</div>",
                    unsafe_allow_html=True)

st.divider()

# ============================================================================
# ADIM 4 — ÇALIŞTIRMA (Molekül Üretimi)
# ============================================================================
st.header("Adım 4 · Çalıştır")

st.markdown(
    "<div class='plain'>ℹ️ Bu adım molekülleri hızlı bir <b>QED (ilaç-benzerlik) "
    "ön skorlaması</b> ile üretir ve eler — reseptör ya da GPU gerekmez, "
    "saniyeler sürer. <b>Gerçek bağlanma docking'i (GNINA, GPU)</b> ayrı bir "
    "adımda, Adım 5'te Google Colab'da yapılır. Böylece Codespaces'te ağır bir "
    "docking motoru çalıştırman gerekmez.</div>",
    unsafe_allow_html=True,
)

if st.button("▶️ Molekülleri Üret ve Skorla", type="primary"):
    if not seeds:
        st.error("En az bir geçerli tohum SMILES gir.")
    elif method == "pretrained":
        st.warning("Pretrained model plugin'i kurulu değil (opsiyonel). "
                   "random / brics / genetic yöntemlerini kullanabilirsin.")
    else:
        # Docking motoru GNINA'ya (Colab, GPU) taşındı — üretim döngüsünde artık
        # yerel docking ÇAĞRILMAZ. Üretim/eleme hızlı QED ön skorlaması ile yapılır;
        # gerçek bağlanma skorları Adım 5'teki GNINA docking'inden gelir.
        docking_opts = None

        if method == "fusion":
            # Füzyonu AYRI süreçte başlat (Sorun 1). Ana thread beklemez;
            # ilerleme aşağıdaki monitör tarafından dosyadan okunarak gösterilir.
            started = launch_fusion(seeds, docking_opts, ss)
            if not started:
                st.info("⚡ Füzyon zaten çalışıyor — mevcut süreç izleniyor.")
            st.rerun()
        else:
            summary = st.empty()
            progress = st.progress(0.0)
            log_box = st.container()
            log_lines = []

            with st.spinner("Moleküller üretiliyor..."):
                if method == "random":
                    mols = mg.random_mutation(seeds, n=int(params["n"]))
                elif method == "brics":
                    mols = mg.brics_recombination(seeds, n=int(params["n"]))
                elif method == "genetic":
                    gen_log = []

                    def log_fn(msg):
                        gen_log.append(msg)
                        with log_box:
                            st.code("\n".join(gen_log[-12:]), language=None)

                    final, mode = mg.genetic_algorithm(
                        seeds,
                        generations=int(params["generations"]),
                        population_size=int(params["population"]),
                        mutation_rate=float(params["mutation_rate"]),
                        docking_opts=docking_opts,
                        log_fn=log_fn,
                    )
                    mols = [s for s, _ in final]

            scored = []
            if method == "genetic":
                for smi, aff in final:
                    scored.append((smi, aff))
                ss["mode"] = mode
            else:
                total = len(mols)
                for i, smi in enumerate(mols):
                    sc, current_mode = mg.score_population([smi], docking_opts)
                    aff = sc.get(smi, 999.0)
                    scored.append((smi, aff))
                    ss["mode"] = current_mode
                    progress.progress((i + 1) / max(total, 1))
                    best = min(a for _, a in scored)
                    summary.markdown(
                        f"<div class='plain'><b>{i+1}/{total}</b> molekül test edildi · "
                        f"şu ana kadar en iyisi <span class='tech'>{best:.3f} kcal/mol</span></div>",
                        unsafe_allow_html=True)
                    log_lines.append(f"[{i+1:>3}/{total}] {aff:>8.3f} kcal/mol   {smi}")
                    with log_box:
                        st.code("\n".join(log_lines[-12:]), language=None)

            progress.progress(1.0)
            scored = [(s, a) for s, a in scored if s and Chem.MolFromSmiles(s)]
            scored.sort(key=lambda x: x[1])
            ss["results"] = scored
            st.success(f"Tamamlandı — {len(scored)} molekül üretildi ve skorlandı.")

# ── Füzyon ilerleme monitörü (ayrı süreç canlıyken her rerun'da çalışır) ──────
if ss.get("fusion_active"):
    render_fusion_monitor(ss)

st.divider()

# ============================================================================
# ADIM 5 — GNINA DOCKING (COLAB, GPU) + PIPELINE (BÖLÜM B + C)
# Docking motoru GNINA'ya taşındı: docking GPU'da, Google Colab'da yapılır.
# Bu adım artık yerel Vina docking'i ÇAĞIRMAZ; yerine kullanıcıyı Colab'a
# yönlendirir, sonucu bekler ve dosya yüklenince pipeline'ın kalanını çalıştırır.
# ============================================================================
COLAB_NB_URL = (
    "https://colab.research.google.com/github/mehmetg06/Remedia/blob/"
    "main/notebooks/gnina_colab.ipynb"
)

st.header("Adım 5 · GNINA Docking (Colab, GPU) + Pipeline")
st.markdown(
    "<div class='plain'>Docking motoru <b>GNINA</b>'ya taşındı ve GPU'da, ücretsiz "
    "Google Colab T4'ünde çalışır. Sıra: molekülleri kaydet → Colab'da GNINA'yı "
    "çalıştır → inen <code>docking_scores.csv</code>'yi bu run klasörüne yükle → "
    "\"Docking Tamamlandı, Devam Et\" butonuna bas. Gerisi (ADMET → sıralama → "
    "dashboard) otomatik.</div>",
    unsafe_allow_html=True,
)

generated_smi = ROOT / "data" / "generated.smi"

# ── ADIM 5.1 · Molekülleri kaydet ve docking için run klasörü hazırla ─────────
if not ss["results"]:
    st.markdown(
        "<div class='plain'>Önce Adım 4'te molekülleri üret ve skorla.</div>",
        unsafe_allow_html=True,
    )
else:
    if st.button("💾 Molekülleri Kaydet & Docking'e Hazırla", type="primary"):
        scores = {s: a for s, a in ss["results"]}
        mg.write_smi([s for s, _ in ss["results"]], generated_smi, scores=scores)

        # Bu docking için sabit bir run_id üret ve klasörünü şimdiden oluştur ki
        # kullanıcı Colab çıktısını TAM olarak nereye yükleyeceğini bilsin.
        run_id = generate_run_id()
        ss["gnina_run_id"] = run_id
        ss["current_run_id"] = run_id
        ss["pipeline_done"] = False

        run_dir = ROOT / "results" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        _shutil.copy2(str(generated_smi), str(run_dir / "input_ligands.smi"))

        st.success(f"Kaydedildi: {generated_smi} · Docking klasörü hazır: results/{run_id}/")

# ── ADIM 5.2 · Colab talimat kutusu + devam butonu ───────────────────────────
run_id = ss.get("gnina_run_id")
if run_id:
    run_dir = ROOT / "results" / run_id
    docking_csv = run_dir / "docking_scores.csv"

    # BÜYÜK, RENKLİ talimat kutusu — dumb-proof adımlar.
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0D1B2A,#10233b);
             border:2px solid {ACCENT}; border-radius:14px; padding:22px 26px;
             margin:14px 0; color:#E2E8F0;">
          <div style="font-size:1.25rem; font-weight:800; color:{ACCENT};
               margin-bottom:12px;">🧬 Sıradaki Adım: GNINA Docking (Colab'da, GPU ile)</div>
          <ol style="font-size:0.98rem; line-height:1.9; margin:0; padding-left:22px;">
            <li>Bu linke tıkla: <a href="{COLAB_NB_URL}" target="_blank"
                style="color:{ACCENT}; font-weight:700;">🔗 Open in Colab (gnina_colab.ipynb)</a></li>
            <li>Colab'da <b>Runtime ▸ Change runtime type ▸ GPU (T4)</b> seç.</li>
            <li>Tüm hücreleri sırayla çalıştır (<b>Shift+Enter</b>).</li>
            <li>İndirilen <code>docking_scores.csv</code>'yi ŞU klasöre yükle:
                <br><code style="color:{ACCENT}; font-size:1.05rem;">results/{run_id}/</code></li>
            <li>Aşağıdaki <b>"Docking Tamamlandı, Devam Et"</b> butonuna bas.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='run-tag'>🆔 Docking çalıştırması: <b>{run_id}</b> · "
        f"Hedef: <b>{ss['uniprot']}</b></div>",
        unsafe_allow_html=True,
    )

    # generated.smi'yi Kopyala — kullanıcı Colab'ın MANUAL_SMILES kutusuna yapıştırabilir.
    with st.expander("📋 generated.smi içeriğini kopyala (Colab'a elle yapıştırmak için)"):
        smi_text = generated_smi.read_text() if generated_smi.exists() else ""
        st.code(smi_text or "(generated.smi henüz yok)", language=None)
        st.markdown(
            "<div class='plain'>İdeal yol: Colab notebook'u repoyu <code>git clone</code> "
            "ettiği için, üretilen molekülleri GitHub'a commit'lersen "
            "<code>Remedia/data/generated.smi</code> otomatik güncel gelir. Commit etmek "
            "istemezsen yukarıdaki metni kopyalayıp Colab ADIM 3'teki "
            "<code>MANUAL_SMILES</code> kutusuna yapıştır.</div>",
            unsafe_allow_html=True,
        )

    # ── Docking Tamamlandı, Devam Et ─────────────────────────────────────────
    if ss["pipeline_running"]:
        st.warning("⏳ Pipeline zaten çalışıyor... Lütfen bekle.")
    elif not tools.get("snakemake", False):
        st.error("❌ Snakemake kurulu değil. Terminalde şunu çalıştır: `bash setup.sh`")
    else:
        if st.button("✅ Docking Tamamlandı, Devam Et", type="primary"):
            if not docking_csv.exists():
                st.error(
                    f"📄 Dosya henüz yüklenmedi: `results/{run_id}/docking_scores.csv`\n\n"
                    "Colab'dan inen `docking_scores.csv`'yi bu klasöre sürükle-bırak ile "
                    "yükle (VS Code / Codespaces dosya gezgininden), sonra tekrar bu "
                    "butona bas."
                )
            else:
                # Yüklenen dosyanın formatını doğrula (ligand,affinity_kcal_mol).
                ok, msg = _validate_docking_csv(docking_csv)
                if not ok:
                    st.error(msg)
                else:
                    st.success(msg)
                    ss["pipeline_running"] = True
                    ss["pipeline_done"] = False

                    # Pipeline'ın kalanını çalıştır: docking_scores.csv zaten mevcut
                    # olduğundan docking kuralı atlanır; ADMET → sıralama → dashboard koşar.
                    cmd = [
                        "snakemake", "--cores", "1",
                        "--config",
                        f"ligands_file={generated_smi}",
                        f"run_id={run_id}",
                        "--rerun-incomplete",
                        "--nolock",
                    ]

                    log_ph = st.empty()
                    status_ph = st.empty()
                    with st.spinner("Pipeline çalışıyor (ADMET → sıralama → dashboard)..."):
                        rc, stdout, stderr = _run_snakemake_live(cmd, log_ph, status_ph)

                    ss["pipeline_running"] = False

                    if rc == 0:
                        ss["pipeline_done"] = True
                        (ROOT / "results" / "latest_run.txt").write_text(run_id)
                        st.success(f"✅ Pipeline tamamlandı! Çalıştırma: **{run_id}**")
                        st.balloons()
                    else:
                        friendly = _interpret_snakemake_error(stderr, stdout)
                        st.error(friendly)
                        with st.expander("🔍 Ham hata detayı"):
                            st.code(stderr[-3000:] if len(stderr) > 3000 else stderr, language=None)

st.divider()

# ============================================================================
# ADIM 6 — SONUÇLAR (Bölüm C: run_id izolasyonu)
# ============================================================================
st.header("Adım 6 · Sonuçlar")

# ── Çalıştırma seçimi (Bölüm C) ──────────────────────────────────────────────
all_runs = get_all_run_ids()
latest_run = get_latest_run_id()

if not all_runs:
    st.markdown(
        "<div class='plain'>Henüz tamamlanmış bir pipeline çalıştırması yok. "
        "Adım 5'te pipeline'ı çalıştır.</div>",
        unsafe_allow_html=True,
    )
else:
    # Geçmiş Çalıştırmalar dropdown
    run_options = all_runs
    default_idx = 0  # En güncel
    if ss.get("current_run_id") and ss["current_run_id"] in run_options:
        default_idx = run_options.index(ss["current_run_id"])

    selected_run = st.selectbox(
        "📂 Çalıştırma seç:",
        options=run_options,
        index=default_idx,
        format_func=lambda r: f"{'⭐ ' if r == latest_run else ''}{r}",
        help="⭐ = en son çalıştırma. Önceki çalıştırmaların sonuçlarına da bakabilirsin.",
    )

    rdir = run_dir_path(selected_run)

    # Çalıştırma etiketi (Bölüm C)
    mol_count = "?"
    target_label = ss.get("uniprot", "?")
    ranking_csv = rdir / "final_ranking.csv"
    if ranking_csv.exists():
        try:
            _rdf = pd.read_csv(ranking_csv)
            mol_count = len(_rdf)
        except Exception:
            pass

    st.markdown(
        f"<div class='run-tag'>🆔 Çalıştırma: <b>{selected_run}</b> · "
        f"<b>{mol_count}</b> molekül · Hedef: <b>{target_label}</b></div>",
        unsafe_allow_html=True,
    )

    # Doğrulama verisi
    _val_csv = rdir / "validated_candidates.csv"
    if not _val_csv.exists():
        _val_csv = ROOT / "results" / "validated_candidates.csv"  # eski konum fallback

    if _val_csv.exists():
        try:
            _val_df = pd.read_csv(_val_csv)
            _cv_csv = rdir / "cross_validated.csv"
            cv_dict = {}
            if _cv_csv.exists():
                _cv_df = pd.read_csv(_cv_csv)
                for _, _cr in _cv_df.iterrows():
                    cv_dict[str(_cr["ligand"]).strip()] = _cr.get("tutarlilik_durumu", "— Sadece Vina test edildi")
            
            for _, _vrow in _val_df.iterrows():
                _lname = str(_vrow.get("ligand", "")).strip()
                _dog = _vrow.get("dogrulanmis_skor")
                _dur = str(_vrow.get("guven_durumu", "")).strip()
                _frk = _vrow.get("fark")
                _tutarlilik = cv_dict.get(_lname, "— Sadece Vina test edildi")
                _sebep = str(_vrow.get("sebep", "") or "").strip()
                if _sebep.lower() in ("nan",):
                    _sebep = ""
                if _lname:
                    ss["validated_data"][_lname] = {
                        "dogrulanmis_skor": float(_dog) if _dog not in ("", None) and str(_dog) not in ("nan", "") else None,
                        "guven_durumu": _dur,
                        "fark": float(_frk) if _frk not in ("", None) and str(_frk) not in ("nan", "") else None,
                        "tutarlilik": _tutarlilik,
                        "sebep": _sebep,
                    }
        except Exception:
            pass

    # ── Docking + ADMET + Ranking sonuçları ──────────────────────────────────
    docking_csv = rdir / "docking_scores.csv"
    admet_csv = rdir / "admet_results.csv"
    dashboard_html = rdir / "dashboard.html"

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        if docking_csv.exists():
            st.markdown("**📊 Docking Skorları**")
            _ddf = pd.read_csv(docking_csv)
            st.dataframe(_ddf, width="stretch", hide_index=True)
            st.download_button("⬇️ İndir", data=_ddf.to_csv(index=False).encode(),
                               file_name=f"{selected_run}_docking.csv", mime="text/csv")
        else:
            st.markdown("<div class='plain'>Docking sonucu yok.</div>", unsafe_allow_html=True)

    with col_r2:
        if admet_csv.exists():
            st.markdown("**🧪 ADMET Filtresi**")
            _adf = pd.read_csv(admet_csv)
            st.dataframe(_adf, width="stretch", hide_index=True)
            st.download_button("⬇️ İndir", data=_adf.to_csv(index=False).encode(),
                               file_name=f"{selected_run}_admet.csv", mime="text/csv")
        else:
            st.markdown("<div class='plain'>ADMET sonucu yok.</div>", unsafe_allow_html=True)

    with col_r3:
        if ranking_csv.exists():
            st.markdown("**🏆 Final Sıralama**")
            _rdf = pd.read_csv(ranking_csv)
            st.dataframe(_rdf, width="stretch", hide_index=True)
            st.download_button("⬇️ İndir", data=_rdf.to_csv(index=False).encode(),
                               file_name=f"{selected_run}_ranking.csv", mime="text/csv")
        else:
            st.markdown("<div class='plain'>Sıralama sonucu yok.</div>", unsafe_allow_html=True)

    # Dashboard linki
    if dashboard_html.exists():
        st.markdown(
            f"<div class='plain'>📋 <b>Dashboard:</b> "
            f"<a href='file://{dashboard_html}' target='_blank'>{dashboard_html.name}</a> "
            f"— bu çalıştırmaya özel raporun tam görünümü.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── UI üretim sonuçları (session_state'ten) ───────────────────────────────
    if ss["results"]:
        results = ss["results"]
        rows = []
        for i, (smi, aff) in enumerate(results):
            props = mol_properties(smi)
            rows.append({
                "rank": i + 1, "name": f"gen_{i:04d}", "SMILES": smi,
                "affinity_kcal_mol": round(aff, 3),
                **props,
            })
        df = pd.DataFrame(rows)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        top = df.iloc[0]
        st.markdown(
            f"<div class='plain'><b>{len(df)}</b> molekül · en iyi affinity "
            f"<span class='tech'>{top['affinity_kcal_mol']} kcal/mol</span> "
            f"(daha negatif = daha güçlü bağlanma).</div>",
            unsafe_allow_html=True,
        )
        st.download_button("⬇️ Ham CSV İndir", data=csv_bytes,
                           file_name="generated_molecules.csv", mime="text/csv")

        st.markdown("### En iyi adaylar")
        for i, (smi, aff) in enumerate(results[:20]):
            props = mol_properties(smi)
            mol_name = f"gen_{i:04d}"
            val_info = ss["validated_data"].get(mol_name) or ss["validated_data"].get(smi[:20])

            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                left, right = st.columns([1, 1.6])
                with left:
                    img = draw_mol(smi)
                    if img is not None:
                        st.image(img, caption=f"#{i+1} · {mol_name}")
                    st.markdown(f"<div class='plain' style='word-break:break-all'>{smi}</div>",
                                unsafe_allow_html=True)

                with right:
                    if val_info:
                        dur = val_info.get("guven_durumu", "")
                        dog_skor = val_info.get("dogrulanmis_skor")
                        fark_val = val_info.get("fark")
                        if dur == "GÜVENİLİR":
                            badge_css = "background:#064E3B;color:#6EE7B7;border:1px solid #059669"
                            badge_icon = "✓"
                            aciklama = "Bu skor daha kapsamlı bir aramayla (yüksek exhaustiveness) doğrulandı."
                        elif "ŞÜPHELİ" in dur:
                            badge_css = "background:#451A03;color:#FCD34D;border:1px solid #D97706"
                            badge_icon = "⚠"
                            aciklama = "Bu skor yüksek exhaustiveness ile tutarsız çıktı — tekrar kontrol edilmesi önerilir."
                        elif "ARTEFAKT" in dur:
                            badge_css = "background:#450A0A;color:#FCA5A5;border:1px solid #DC2626"
                            badge_icon = "✗"
                            aciklama = "Bu skor ilk taramada artefakt olarak işaretlendi."
                        else:
                            # DOĞRULANAMADI vb. — GERÇEK sebebi göster (genel mesaj değil).
                            badge_css = "background:#1E293B;color:#94A3B8;border:1px solid #334155"
                            badge_icon = "?"
                            _sebep = str(val_info.get("sebep", "") or "").strip()
                            if _sebep:
                                aciklama = f"Doğrulama tamamlanamadı — sebep: {_sebep}"
                            else:
                                aciklama = "Doğrulama tamamlanamadı."

                        tutarlilik = val_info.get("tutarlilik", "— Sadece Vina test edildi")
                        if "TUTARLI" in tutarlilik and "TUTARSIZ" not in tutarlilik:
                            tut_css = "background:#064E3B;color:#6EE7B7;border:1px solid #059669"
                            tut_icon = "✓ İki motor da onaylıyor"
                        elif "TUTARSIZ" in tutarlilik:
                            tut_css = "background:#451A03;color:#FCD34D;border:1px solid #D97706"
                            tut_icon = "⚠ Motorlar anlaşmıyor"
                        else:
                            tut_css = "background:#1E293B;color:#94A3B8;border:1px solid #334155"
                            tut_icon = "— Sadece Vina test edildi"

                        st.markdown(
                            f"""<div style='margin-bottom:10px'>
                            <span style='padding:4px 12px;border-radius:20px;font-size:0.78rem;
                                font-weight:700;font-family:monospace;{badge_css}'>
                                {badge_icon} Doğrulama: {dur}
                            </span>
                            <span style='padding:4px 12px;border-radius:20px;font-size:0.78rem;
                                font-weight:700;font-family:monospace;{tut_css}; margin-left:8px'>
                                {tut_icon}
                            </span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        sc1, sc2, sc3 = st.columns(3)
                        
                        mode_badge = "<span style='background:#064E3B;color:#6EE7B7;padding:2px 6px;border-radius:4px;font-size:0.7rem;'>✓ Gerçek Docking Kullanıldı</span>" if ss.get("mode") == "real_docking" else "<span style='background:#7F1D1D;color:#FCA5A5;padding:2px 6px;border-radius:4px;font-size:0.7rem;'>⚠ Yedek QED Skoru — Reseptör Verilmedi</span>"
                        sc1.markdown(f"**İlk Skor**<br><span class='tech' style='font-size:1.5rem'>{aff:.2f}</span><br>{mode_badge}", unsafe_allow_html=True)
                        if dog_skor is not None:
                            delta_str = f"{fark_val:+.2f}" if fark_val is not None else ""
                            sc2.metric("Doğrulanmış Skor", f"{dog_skor:.2f}", delta=delta_str, delta_color="inverse")
                        sc3.metric("MW", props.get("MW", "-"))
                        st.markdown(f"<div class='plain' style='font-size:0.8rem;margin-top:4px'>ℹ️ {aciklama}</div>",
                                    unsafe_allow_html=True)
                    else:
                        m1, m2, m3 = st.columns(3)
                        mode_badge = "<span style='background:#064E3B;color:#6EE7B7;padding:2px 6px;border-radius:4px;font-size:0.7rem;'>✓ Gerçek Docking Kullanıldı</span>" if ss.get("mode") == "real_docking" else "<span style='background:#7F1D1D;color:#FCA5A5;padding:2px 6px;border-radius:4px;font-size:0.7rem;'>⚠ Yedek QED Skoru — Reseptör Verilmedi</span>"
                        m1.markdown(f"**Affinity (kcal/mol)**<br><span class='tech' style='font-size:1.5rem'>{aff:.2f}</span><br>{mode_badge}", unsafe_allow_html=True)
                        m1.metric("MW", props.get("MW", "-"))
                        m2.metric("LogP", props.get("LogP", "-"))
                        m2.metric("TPSA", props.get("TPSA", "-"))
                        m3.metric("HBD / HBA", f"{props.get('HBD','-')} / {props.get('HBA','-')}")
                        m3.metric("QED", props.get("QED", "-"))
                        st.markdown(
                            "<div class='plain' style='font-size:0.78rem;color:#7E8C9A;margin-top:6px'>"
                            "ℹ️ Bu skor ilk taramadan geliyor, henüz kapsamlı doğrulama yapılmadı — "
                            "yanıltıcı olabilir. Pipeline'ı çalıştırarak tam docking yap."
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"<div class='plain'>💬 {interpret(smi, aff, props)}</div>",
                                unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='plain'>Henüz sonuç yok. Adım 4'te 'Üret ve Skorla' butonuna bas.</div>",
            unsafe_allow_html=True,
        )
