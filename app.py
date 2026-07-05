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

5 adım: (1) Hedef seçimi → (2) Pocket seçimi → (3) Üretim yöntemi →
        (4) Çalıştırma → (5) Sonuçlar
"""
import sys
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

# dashboard.html paletiyle uyumlu ince rötuşlar (config.toml zaten temeli veriyor).
ACCENT = "#5EEAD4"
st.markdown(
    f"""
    <style>
      .plain {{ color:#7E8C9A; font-size:0.85rem; line-height:1.45; }}
      .tech  {{ color:{ACCENT}; font-family:'IBM Plex Mono',monospace; font-weight:600; }}
      .card  {{ background:#10161D; border:1px solid rgba(255,255,255,0.07);
                border-radius:12px; padding:16px 18px; margin-bottom:12px; }}
      .eyebrow {{ color:{ACCENT}; font-family:monospace; letter-spacing:0.15em;
                  font-size:0.72rem; text-transform:uppercase; }}
      h1,h2,h3 {{ font-family:'Space Grotesk',sans-serif; }}
      .stProgress > div > div > div > div {{ background:{ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def dual(label, value, plain, unit=""):
    """Bir alanın [teknik etiket + ham değer] ile [sade açıklama]'sını yan yana
    gösterir — projenin çekirdek tasarım prensibi."""
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
# YARDIMCI: PDB analiz (pLDDT, atom sayısı, boyut)
# ============================================================================
def analyze_pdb(pdb_path: Path) -> dict:
    """AlphaFold PDB'sinden ham metrikleri çıkarır: dosya boyutu, atom sayısı,
    ortalama pLDDT (B-factor kolonu)."""
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
# YARDIMCI: molekül özellikleri + skorlama
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
    """Ham verilerin sade dille yorumu."""
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
st.divider()

# Oturum durumu
ss = st.session_state
ss.setdefault("pdb_info", None)
ss.setdefault("uniprot", "P30405")
ss.setdefault("pocket", None)
ss.setdefault("results", None)

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
    try:
        with st.spinner(f"{uniprot} yapısı AlphaFold DB'den indiriliyor..."):
            import fetch_structure
            pdb_path = fetch_structure.fetch_alphafold(uniprot)
            ss["pdb_info"] = {"path": str(pdb_path), **analyze_pdb(pdb_path)}
        st.success(f"İndirildi: {pdb_path}")
    except Exception as e:
        st.error(f"İndirme başarısız: {e}")

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

# config.yaml'daki cep bilgisini varsayılan olarak yükle (fpocket yoksa da çalışsın).
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
# ADIM 3 — ÜRETİM YÖNTEMİ
# ============================================================================
st.header("Adım 3 · Molekül Üretim Yöntemi")
st.markdown(
    "<div class='plain'>Yeni moleküllerin nasıl türetileceğini seç. Her yöntemin "
    "parametreleri varsayılan değerleriyle <b>doğrudan görünür</b> — istersen değiştir, "
    "istemezsen olduğu gibi bırak.</div>",
    unsafe_allow_html=True,
)

seeds_text = st.text_area(
    "Tohum molekül(ler) — her satıra bir SMILES",
    value="CC(=O)Oc1ccccc1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O",
    height=90,
    help="Üretim bu 'başlangıç' moleküllerinden yola çıkar. Örnekler: aspirin, ibuprofen.",
)
seeds = [s.strip().split()[0] for s in seeds_text.splitlines() if s.strip() and not s.startswith("#")]

method = st.radio(
    "Yöntem",
    ["random", "brics", "genetic", "pretrained"],
    format_func=lambda m: {
        "random": "🎲 Random Mutation",
        "brics": "🧩 BRICS Fragment Recombination",
        "genetic": "🧬 Genetic Algorithm",
        "pretrained": "🤖 Pretrained Model (opsiyonel)",
    }[m],
)

col_a, col_b = st.columns([1, 1.4])
params = {}
if method == "random":
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
                                            help="Yeni bireylerin ne kadarı mutasyonla (çaprazlama yerine) üretilsin.")
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
# ADIM 4 — ÇALIŞTIRMA
# ============================================================================
st.header("Adım 4 · Çalıştır")

# Reseptör PDBQT (gerçek docking için) var mı?
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

if st.button("▶️ Molekülleri Üret ve Skorla", type="primary"):
    if not seeds:
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

        # --- Üret ---
        with st.spinner("Moleküller üretiliyor..."):
            if method == "random":
                mols = mg.random_mutation(seeds, n=int(params["n"]))
            elif method == "brics":
                mols = mg.brics_recombination(seeds, n=int(params["n"]))
            else:  # genetic — log akışıyla
                gen_log = []

                def log_fn(msg):
                    gen_log.append(msg)
                    with log_box:
                        st.code("\n".join(gen_log[-12:]), language=None)

                final = mg.genetic_algorithm(
                    seeds,
                    generations=int(params["generations"]),
                    population_size=int(params["population"]),
                    mutation_rate=float(params["mutation_rate"]),
                    docking_opts=docking_opts,
                    log_fn=log_fn,
                )
                mols = [s for s, _ in final]

        # --- Skorla (random/brics için; genetic zaten skorlanmış son popülasyon döndürür) ---
        scored = []
        if method == "genetic":
            for smi, aff in final:
                scored.append((smi, aff))
        else:
            total = len(mols)
            for i, smi in enumerate(mols):
                sc = mg.score_population([smi], docking_opts)
                aff = sc.get(smi, 999.0)
                scored.append((smi, aff))
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
        # En iyiden kötüye sırala
        scored = [(s, a) for s, a in scored if s and Chem.MolFromSmiles(s)]
        scored.sort(key=lambda x: x[1])
        ss["results"] = scored
        st.success(f"Tamamlandı — {len(scored)} molekül üretildi ve skorlandı.")

st.divider()

# ============================================================================
# ADIM 5 — SONUÇLAR
# ============================================================================
st.header("Adım 5 · Sonuçlar")

if not ss["results"]:
    st.markdown("<div class='plain'>Henüz sonuç yok. Adım 4'te 'Üret ve Skorla' butonuna bas.</div>",
                unsafe_allow_html=True)
else:
    results = ss["results"]

    # --- Ham CSV (HER ZAMAN görünür) ---
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
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            left, right = st.columns([1, 1.6])
            with left:
                img = draw_mol(smi)
                if img is not None:
                    st.image(img, caption=f"#{i+1} · gen_{i:04d}")
                st.markdown(f"<div class='plain' style='word-break:break-all'>{smi}</div>",
                            unsafe_allow_html=True)
            with right:
                # Sağ üst: ham teknik veriler
                m1, m2, m3 = st.columns(3)
                m1.metric("Affinity (kcal/mol)", f"{aff:.2f}")
                m1.metric("MW", props.get("MW", "-"))
                m2.metric("LogP", props.get("LogP", "-"))
                m2.metric("TPSA", props.get("TPSA", "-"))
                m3.metric("HBD / HBA", f"{props.get('HBD','-')} / {props.get('HBA','-')}")
                m3.metric("QED", props.get("QED", "-"))
                # Sağ alt: sade yorum
                st.markdown(f"<div class='plain'>💬 {interpret(smi, aff, props)}</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Pipeline'a aktarım ---
    st.markdown("### Pipeline'a aktar")
    st.markdown(
        "<div class='plain'>Bu moleküller mevcut docking + ADMET + sıralama zincirine "
        "doğrudan girebilir. Aşağıdaki butonla <code>data/generated.smi</code> olarak kaydet, "
        "sonra <code>snakemake --cores 1 --config ligands_file=data/generated.smi</code> "
        "çalıştır.</div>",
        unsafe_allow_html=True,
    )
    if st.button("💾 data/generated.smi olarak kaydet"):
        out = ROOT / "data" / "generated.smi"
        scores = {s: a for s, a in results}
        mg.write_smi([s for s, _ in results], out, scores=scores)
        st.success(f"Kaydedildi: {out}")
