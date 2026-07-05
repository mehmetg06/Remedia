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
import io
import os
import shutil
import subprocess
import sys
import threading
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


@st.cache_data(ttl=60)
def check_tools() -> dict[str, bool]:
    return {
        "snakemake": _check_tool("snakemake"),
        "vina": _check_tool("vina") or _check_tool("autodock_vina"),
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

def _interpret_snakemake_error(stderr: str, stdout: str) -> str:
    """Ham Snakemake hata çıktısını sade Türkçe'ye çevirir."""
    combined = (stderr + stdout).lower()
    if "vina" in combined and ("not found" in combined or "command not found" in combined):
        return "🔴 Docking adımında hata: AutoDock Vina kurulu değil. `conda install -c conda-forge autodock-vina` komutuyla kurabilirsin."
    if "obabel" in combined and "not found" in combined:
        return "🔴 Ligand hazırlama adımında hata: Open Babel (obabel) kurulu değil. `apt install openbabel` ile kurabilirsin."
    if "fpocket" in combined and "not found" in combined:
        return "🔴 Pocket tespitinde hata: fpocket kurulu değil. https://github.com/Discngine/fpocket adresinden kurabilirsin."
    if "modulenotfounderror" in combined or "importerror" in combined:
        return "🔴 Python modülü eksik. `pip install -r requirements.txt` komutunu çalıştır."
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
    cols = st.columns(4)
    labels = {"snakemake": "Snakemake", "vina": "AutoDock Vina", "fpocket": "fpocket", "obabel": "Open Babel"}
    install_hints = {
        "snakemake": "pip install snakemake --break-system-packages",
        "vina": "conda install -c conda-forge autodock-vina",
        "fpocket": "https://github.com/Discngine/fpocket",
        "obabel": "apt install openbabel",
    }
    for (k, v), col in zip(tools.items(), cols):
        icon = "✅" if v else "❌"
        css = "status-ok" if v else "status-err"
        col.markdown(
            f"<span class='{css}'>{icon} {labels[k]}</span>"
            + (f"<br><span style='font-size:0.72rem;color:#7E8C9A'>{install_hints[k]}</span>" if not v else ""),
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
ss.setdefault("pipeline_done", False)         # Bölüm B

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

    # ── BÖLÜM A: Bilinen ligandları otomatik çek ──────────────────────────
    if ss["pdb_info"]:
        with st.spinner(f"🔍 {uniprot} için bilinen inhibitörler ChEMBL'de aranıyor..."):
            try:
                from known_ligands import fetch_known_ligands
                ligands, msg = fetch_known_ligands(uniprot, max_results=5)
                ss["known_ligands"] = ligands
                ss["known_ligands_msg"] = msg
            except Exception as exc:
                ss["known_ligands"] = []
                ss["known_ligands_msg"] = f"⚠️ Bilinen ligand araması başarısız: {exc}"

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

try:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
except Exception:
    cfg = {}

pk = (cfg.get("dashboard", {}) or {}).get("pocket", {})
center = cfg.get("pocket_center", [5.00, -1.02, -15.56])
box = cfg.get("box_size", [20.0, 20.0, 20.0])

import pandas as pd
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
    use_container_width=True,
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
                st.image(png, use_container_width=True)
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

    default_seeds = "\n".join(selected_smiles) if selected_smiles else "CC(=O)Oc1ccccc1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
else:
    # Hiç bulunamadı
    if known_msg:
        st.info(known_msg)
    default_seeds = "CC(=O)Oc1ccccc1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
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

receptor_pdbqt = ROOT / "data" / f"{ss['uniprot']}_alphafold.pdbqt"
use_real_docking = receptor_pdbqt.exists()
if use_real_docking:
    st.markdown(f"<div class='plain'>✅ Reseptör PDBQT bulundu — <b>gerçek AutoDock Vina</b> "
                f"skorlaması kullanılacak.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='plain'>ℹ️ Reseptör PDBQT (Vina girdisi) bulunamadı — skorlama "
                "<b>QED tabanlı yedek fitness</b> ile yapılacak (Vina kurulumu tamamlanınca "
                "otomatik gerçek docking'e geçer). Üretim ve elemeler yine tam çalışır.</div>",
                unsafe_allow_html=True)

qed_fallback_consent = True
if not use_real_docking and method in ["genetic", "fusion"]:
    qed_fallback_consent = st.checkbox("Reseptör seçilmedi — sonuçlar gerçek bağlanma tahmini DEĞİL, sadece ilaç-benzerlik puanı olacak. Devam?", value=False)

if st.button("▶️ Molekülleri Üret ve Skorla", type="primary"):
    if not use_real_docking and method in ["genetic", "fusion"] and not qed_fallback_consent:
        st.error("Lütfen yukarıdaki uyarıyı onaylayın veya Adım 1'den bir reseptör seçin.")
    elif not seeds:
        st.error("En az bir geçerli tohum SMILES gir.")
    elif method == "pretrained":
        st.warning("Pretrained model plugin'i kurulu değil (opsiyonel). "
                   "random / brics / genetic yöntemlerini kullanabilirsin.")
    else:
        summary = st.empty()
        progress = st.progress(0.0)
        log_box = st.container()
        log_lines = []

        docking_opts = None
        if use_real_docking and ss["pocket"]:
            docking_opts = {
                "receptor": str(receptor_pdbqt),
                "center": ss["pocket"]["center"],
                "box_size": ss["pocket"]["box"],
                "workdir": str(ROOT / "results" / "ui_ga_work"),
                "exhaustiveness": cfg.get("exhaustiveness", 8),
            }

        with st.spinner("Moleküller üretiliyor..."):
            if method == "random":
                mols = mg.random_mutation(seeds, n=int(params["n"]))
            elif method == "brics":
                mols = mg.brics_recombination(seeds, n=int(params["n"]))
            else:
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
            elif method == "fusion":
                gen_log = []

                def log_fn(msg):
                    gen_log.append(msg)
                    with log_box:
                        st.code("\n".join(gen_log[-12:]), language=None)
                        
                final, mode = mg.fusion_generation(
                    seeds,
                    docking_opts=docking_opts,
                    log_fn=log_fn,
                )
                mols = [s for s, _ in final]

        scored = []
        if method in ["genetic", "fusion"]:
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

st.divider()

# ============================================================================
# ADIM 5 — PIPELINE (BÖLÜM B + C)
# ============================================================================
st.header("Adım 5 · Full Pipeline'ı Çalıştır")
st.markdown(
    "<div class='plain'>Üretilen molekülleri <b>Snakemake pipeline'ından</b> geçir: "
    "ligand hazırlama → docking → ADMET filtresi → sıralama → dashboard. "
    "Terminal açmana gerek yok — tek butonla çalışır.</div>",
    unsafe_allow_html=True,
)

# ── Pipeline tetikleme ────────────────────────────────────────────────────────
generated_smi = ROOT / "data" / "generated.smi"

# "Kaydet + Pipeline Çalıştır" butonu
col_save, col_run = st.columns([1, 2])

with col_save:
    if ss["results"]:
        if st.button("💾 Molekülleri Kaydet (data/generated.smi)"):
            scores = {s: a for s, a in ss["results"]}
            mg.write_smi([s for s, _ in ss["results"]], generated_smi, scores=scores)
            st.success(f"Kaydedildi: {generated_smi}")
    else:
        st.markdown(
            "<div class='plain'>Önce Adım 4'te molekülleri üret ve skorla.</div>",
            unsafe_allow_html=True,
        )

with col_run:
    # Çift-çalıştırma önlemi (Bölüm B)
    if ss["pipeline_running"]:
        st.warning("⏳ Pipeline zaten çalışıyor... Lütfen bekle.")
    elif not tools.get("snakemake", False):
        st.error(
            "❌ Snakemake kurulu değil. Kurulum için:\n"
            "`pip install snakemake --break-system-packages`"
        )
    else:
        btn_label = "🚀 Pipeline'ı Çalıştır (Snakemake)"
        if st.button(btn_label, type="primary", disabled=not generated_smi.exists()):
            if not generated_smi.exists():
                st.error("Önce molekülleri kaydet (sol taraftaki butona bas).")
            else:
                # run_id üret (Bölüm C)
                run_id = generate_run_id()
                ss["current_run_id"] = run_id
                ss["pipeline_running"] = True
                ss["pipeline_done"] = False

                run_dir = ROOT / "results" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                # Ligand dosyasını run klasörüne kopyala
                import shutil as _shutil
                _shutil.copy2(str(generated_smi), str(run_dir / "input_ligands.smi"))

                cmd = [
                    "snakemake", "--cores", "1",
                    "--config",
                    f"ligands_file={generated_smi}",
                    f"run_id={run_id}",
                    "--rerun-incomplete",
                    "--nolock",
                ]

                st.markdown(
                    f"<div class='run-tag'>🆔 Çalıştırma: <b>{run_id}</b> · "
                    f"Hedef: <b>{ss['uniprot']}</b></div>",
                    unsafe_allow_html=True,
                )

                log_ph = st.empty()
                status_ph = st.empty()

                with st.spinner("Pipeline çalışıyor..."):
                    rc, stdout, stderr = _run_snakemake_live(cmd, log_ph, status_ph)

                ss["pipeline_running"] = False

                if rc == 0:
                    ss["pipeline_done"] = True
                    # latest_run.txt güncelle
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
                if _lname:
                    ss["validated_data"][_lname] = {
                        "dogrulanmis_skor": float(_dog) if _dog not in ("", None) and str(_dog) not in ("nan", "") else None,
                        "guven_durumu": _dur,
                        "fark": float(_frk) if _frk not in ("", None) and str(_frk) not in ("nan", "") else None,
                        "tutarlilik": _tutarlilik,
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
            st.dataframe(_ddf, use_container_width=True, hide_index=True)
            st.download_button("⬇️ İndir", data=_ddf.to_csv(index=False).encode(),
                               file_name=f"{selected_run}_docking.csv", mime="text/csv")
        else:
            st.markdown("<div class='plain'>Docking sonucu yok.</div>", unsafe_allow_html=True)

    with col_r2:
        if admet_csv.exists():
            st.markdown("**🧪 ADMET Filtresi**")
            _adf = pd.read_csv(admet_csv)
            st.dataframe(_adf, use_container_width=True, hide_index=True)
            st.download_button("⬇️ İndir", data=_adf.to_csv(index=False).encode(),
                               file_name=f"{selected_run}_admet.csv", mime="text/csv")
        else:
            st.markdown("<div class='plain'>ADMET sonucu yok.</div>", unsafe_allow_html=True)

    with col_r3:
        if ranking_csv.exists():
            st.markdown("**🏆 Final Sıralama**")
            _rdf = pd.read_csv(ranking_csv)
            st.dataframe(_rdf, use_container_width=True, hide_index=True)
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
                            badge_css = "background:#1E293B;color:#94A3B8;border:1px solid #334155"
                            badge_icon = "?"
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
