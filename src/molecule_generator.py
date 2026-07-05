# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
molecule_generator.py
Sıfırdan YENİ aday molekül ÜRETEN katman — model eğitimi GEREKTİRMEZ, yalnızca
RDKit ile kimyasal kurallara dayalı yöntemler kullanır.

Dört üretim yöntemi:
    a) random   — Rastgele mutasyon (atom değişimi + fonksiyonel grup ekle/çıkar)
    b) brics     — BRICS fragment rekombinasyonu (moleküllerin parçalarını birleştir)
    c) genetic   — Genetik algoritma (her nesilde docking ile skorla, en iyileri üret)
    d) pretrained — REINVENT vb. hazır model için opsiyonel plugin (stub)

Çıktı: mevcut pipeline'a DOĞRUDAN girebilen bir .smi dosyası
       (format: "SMILES  isim"), böylece ligand_prep → docking → admet → rank
       zinciri hiçbir değişiklik gerektirmeden çalışır.

Kullanım örnekleri:
    # Rastgele mutasyon
    python src/molecule_generator.py --method random \
        --seeds "CC(=O)Oc1ccccc1C(=O)O" --n 50 --output data/generated.smi

    # BRICS rekombinasyonu (birden fazla tohum)
    python src/molecule_generator.py --method brics \
        --seeds "CC(=O)Oc1ccccc1C(=O)O" "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
        --n 50 --output data/generated.smi

    # Genetik algoritma (docking ile skorlama)
    python src/molecule_generator.py --method genetic \
        --seeds-file data/ligands_example.smi \
        --generations 10 --population 30 \
        --receptor data/P30405_alphafold.pdbqt \
        --center 5.00 -1.02 -15.56 --size 20 20 20 \
        --output data/generated.smi
"""
import argparse
import random
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors, QED

# RDKit'in sanitization uyarılarını sustur — geçersiz varyantları zaten
# kendimiz eleyeceğiz, log'u kirletmesinler.
RDLogger.DisableLog("rdApp.*")

# --- Mutasyon "yapı taşları" -------------------------------------------------
# Atom değişimi için kullanılacak elementler (organik kimyada yaygın, valansı
# uyumlu heteroatomlar): C, N, O, S
SWAP_ATOMS = [6, 7, 8, 16]

# Eklenebilecek küçük fonksiyonel gruplar — SMILES parçası olarak.
# (isim -> ekli grubun SMILES'i; * bağlanma noktası)
FUNCTIONAL_GROUPS = {
    "metil": "C",
    "hidroksil": "O",
    "amin": "N",
    "flor": "F",
    "klor": "Cl",
    "brom": "Br",
    "karbonil": "C=O",
}


# ============================================================================
# ORTAK YARDIMCILAR
# ============================================================================
def canonical_or_none(mol) -> str | None:
    """Bir mol nesnesini sanitize edip kanonik SMILES döndürür; kimyasal olarak
    imkânsız/geçersizse None döner (böylece otomatik elenir)."""
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
        # Tekrar parse ederek gerçekten geçerli olduğunu doğrula.
        reparsed = Chem.MolFromSmiles(smi)
        if reparsed is None:
            return None
        return Chem.MolToSmiles(reparsed)
    except Exception:
        return None


def is_reasonable(smi: str) -> bool:
    """Üretilen molekülü basit "ilaç-benzeri" sınırlarla eler: aşırı küçük ya da
    devasa moleküller ıskartaya çıkar (docking'i boşa yormasınlar)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 4 or n_heavy > 60:
        return False
    mw = Descriptors.MolWt(mol)
    if mw > 900:
        return False
    return True


# ============================================================================
# a) RANDOM MUTATION
# ============================================================================
def _swap_atom(mol):
    """Rastgele bir ağır atomu başka bir elementle (C/N/O/S) değiştirir."""
    rwmol = Chem.RWMol(mol)
    candidates = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetAtomicNum() in SWAP_ATOMS]
    if not candidates:
        return None
    idx = random.choice(candidates)
    atom = rwmol.GetAtomWithIdx(idx)
    new_num = random.choice([z for z in SWAP_ATOMS if z != atom.GetAtomicNum()])
    atom.SetAtomicNum(new_num)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    atom.SetFormalCharge(0)
    return rwmol.GetMol()


def _add_group(mol):
    """Boş valansı olan bir atoma küçük bir fonksiyonel grup ekler."""
    # Yeterli implicit H'ı olan (yani bağ ekleyebileceğimiz) bir atom seç.
    hosts = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
    if not hosts:
        return None
    host_idx = random.choice(hosts)
    group_smi = random.choice(list(FUNCTIONAL_GROUPS.values()))
    frag = Chem.MolFromSmiles(group_smi)
    if frag is None:
        return None
    # Fragmanı ana molekülle birleştir, ilk atomunu host'a tek bağla bağla.
    n_before = mol.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))
    combo.AddBond(host_idx, n_before, Chem.BondType.SINGLE)  # frag'ın ilk atomu
    return combo.GetMol()


def _remove_atom(mol):
    """Terminal (tek bağlı, halka dışı) bir atomu siler — grup çıkarma."""
    terminals = [
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetDegree() == 1 and not a.IsInRing()
    ]
    if len(terminals) <= 1:  # tümüyle boşaltma riski, atla
        return None
    rwmol = Chem.RWMol(mol)
    rwmol.RemoveAtom(random.choice(terminals))
    return rwmol.GetMol()


_MUTATION_OPS = [_swap_atom, _add_group, _remove_atom]


def mutate_once(smiles: str) -> str | None:
    """Bir tohum SMILES'e tek bir rastgele mutasyon uygular; sonuç geçerli
    değilse None döner."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    op = random.choice(_MUTATION_OPS)
    try:
        new_mol = op(mol)
    except Exception:
        return None
    if new_mol is None:
        return None
    return canonical_or_none(new_mol)


def random_mutation(seeds: list[str], n: int, max_tries_factor: int = 40) -> list[str]:
    """Tohum moleküllere rastgele mutasyonlar uygulayarak `n` benzersiz, geçerli
    varyant üretir. Geçersiz/imkânsız moleküller RDKit sanitization ile elenir."""
    results: set[str] = set()
    seed_set = {s for s in seeds if Chem.MolFromSmiles(s) is not None}
    if not seed_set:
        return []
    tries = 0
    max_tries = n * max_tries_factor
    while len(results) < n and tries < max_tries:
        tries += 1
        seed = random.choice(list(seed_set))
        # 1-3 ardışık mutasyon zinciri — biraz daha çeşitlilik.
        current = seed
        for _ in range(random.randint(1, 3)):
            nxt = mutate_once(current)
            if nxt is None:
                break
            current = nxt
        if current and current not in seed_set and is_reasonable(current):
            results.add(current)
    return list(results)


# ============================================================================
# b) BRICS / RECAP FRAGMENT REKOMBİNASYONU
# ============================================================================
def brics_recombination(seeds: list[str], n: int, max_tries_factor: int = 30) -> list[str]:
    """Tohum molekülleri BRICS kurallarıyla fragmanlarına ayırır ve fragmanları
    çapraz birleştirerek yeni moleküller kurar."""
    frags: set[str] = set()
    for s in seeds:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        # BRICSDecompose parçalayamazsa molekülün kendisini fragman say.
        pieces = set(BRICS.BRICSDecompose(mol))
        frags.update(pieces if pieces else {Chem.MolToSmiles(mol)})

    frag_mols = [Chem.MolFromSmiles(f) for f in frags]
    frag_mols = [m for m in frag_mols if m is not None]
    if len(frag_mols) < 2:
        return []

    results: set[str] = set()
    seed_set = {Chem.MolToSmiles(Chem.MolFromSmiles(s))
                for s in seeds if Chem.MolFromSmiles(s) is not None}

    # BRICSBuild bir jeneratördür; fragman havuzunu karıştırıp örnekleriz.
    max_tries = n * max_tries_factor
    tries = 0
    while len(results) < n and tries < max_tries:
        tries += 1
        random.shuffle(frag_mols)
        try:
            builder = BRICS.BRICSBuild(frag_mols, maxDepth=random.randint(1, 3))
            for prod in builder:
                tries += 1
                try:
                    prod.UpdatePropertyCache(strict=False)
                    smi = canonical_or_none(prod)
                except Exception:
                    smi = None
                if smi and smi not in seed_set and is_reasonable(smi):
                    results.add(smi)
                if len(results) >= n or tries >= max_tries:
                    break
        except Exception:
            continue
    return list(results)


# ============================================================================
# c) GENETİK ALGORİTMA
# ============================================================================
def crossover(smi1: str, smi2: str) -> str | None:
    """İki molekülün BRICS fragmanlarını birleştirerek bir "yavru" üretir."""
    children = brics_recombination([smi1, smi2], n=1, max_tries_factor=15)
    return children[0] if children else None


def _pseudo_affinity(smi: str) -> float:
    """Vina / reseptör mevcut DEĞİLKEN kullanılan yedek fitness — QED (ilaç-benzerlik)
    tabanlı bir vekil skor. Daha negatif = daha iyi (docking affinity ile aynı yön).
    Not: Bu gerçek docking'in yerini TUTMAZ, yalnızca modül reseptörsüz de test
    edilebilsin diye vardır."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    try:
        qed = QED.qed(mol)
    except Exception:
        qed = 0.0
    # QED [0,1] -> yaklaşık [-12, 0] kcal/mol aralığına ölçekle (kabaca gerçekçi).
    return -(2.0 + qed * 10.0)


def score_population(smiles_list: list[str], docking_opts: dict | None) -> dict[str, float]:
    """Popülasyondaki her molekülü skorlar.

    docking_opts verilmişse (reseptör + kutu) gerçek pipeline'ı çalıştırır:
        SMILES → ligand_prep (3D/PDBQT) → docking.dock_all (Vina).
    Aksi halde QED tabanlı yedek fitness'e düşer, böylece Vina kurulu olmadan da
    algoritma çalışır.

    Döndürür: {SMILES: affinity_kcal_mol}. Skorlanamayan moleküller büyük (kötü)
    bir ceza değeriyle işaretlenir ki elenme eğiliminde olsunlar.
    """
    if not docking_opts:
        return {smi: _pseudo_affinity(smi) for smi in smiles_list}

    scores = _dock_smiles(
        smiles_list,
        receptor=docking_opts["receptor"],
        center=docking_opts["center"],
        box_size=docking_opts["box_size"],
        workdir=docking_opts["workdir"],
        exhaustiveness=docking_opts.get("exhaustiveness", 8),
    )
    # Docking başarısız olanlara ceza (elenmeleri için).
    return {smi: (scores.get(smi) if scores.get(smi) is not None else 999.0)
            for smi in smiles_list}


def _dock_smiles(smiles_list, receptor, center, box_size, workdir, exhaustiveness=8):
    """SMILES listesini gerçek Vina docking'inden geçirir ve {SMILES: affinity}
    döndürür. ligand_prep ve docking modüllerini fonksiyon olarak import eder
    (CLI bozulmaz). Vina/meeko kurulu değilse yedek fitness'e düşer."""
    try:
        # Aynı klasördeki modülleri import et.
        import importlib.util
        import sys
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        import ligand_prep  # noqa: E402
        import docking  # noqa: E402
    except Exception as e:
        print(f"[UYARI] Docking modülleri yüklenemedi ({e}); yedek fitness kullanılıyor.")
        return {smi: _pseudo_affinity(smi) for smi in smiles_list}

    workdir = Path(workdir)
    prepared = workdir / "prepared"
    poses = workdir / "poses"
    prepared.mkdir(parents=True, exist_ok=True)

    # Her SMILES'e kararlı bir isim ver ve 3D/PDBQT hazırla.
    name_to_smi = {}
    for i, smi in enumerate(smiles_list):
        name = f"gen_{i:04d}"
        name_to_smi[name] = smi
        try:
            sdf = ligand_prep.prepare_ligand(smi, name, prepared)
            if sdf is not None:
                ligand_prep.convert_to_pdbqt(sdf)
        except Exception as e:
            print(f"[UYARI] {name} hazırlanamadı: {e}")

    try:
        results = docking.dock_all(
            Path(receptor), prepared, center=list(center), box_size=list(box_size),
            poses_dir=poses, exhaustiveness=exhaustiveness,
        )
    except Exception as e:
        print(f"[UYARI] Vina docking çalışmadı ({e}); yedek fitness kullanılıyor.")
        return {smi: _pseudo_affinity(smi) for smi in smiles_list}

    smi_scores = {}
    for r in results:
        smi = name_to_smi.get(r["ligand"])
        if smi is not None:
            smi_scores[smi] = r["affinity_kcal_mol"]
    return smi_scores


def genetic_algorithm(
    seeds: list[str],
    generations: int = 10,
    population_size: int = 30,
    elite_frac: float = 0.20,
    mutation_rate: float = 0.30,
    docking_opts: dict | None = None,
    log_fn=print,
) -> list[tuple[str, float]]:
    """Genetik algoritma ile molekül optimizasyonu.

    - Başlangıç popülasyonu: tohumlar + random mutasyon + BRICS ürünleri.
    - Her nesilde: popülasyonu skorla (fitness = -affinity → daha negatif daha iyi),
      en iyi %`elite_frac`'i tut, gerisini çaprazlama + mutasyonla yeniden üret.
    - Her neslin en iyi molekülünü ve skorunu loglar.

    Döndürür: son popülasyonu (SMILES, affinity) çiftleri olarak, en iyiden kötüye
    sıralı.
    """
    # --- Başlangıç popülasyonu ---
    seed_set = [s for s in seeds if Chem.MolFromSmiles(s) is not None]
    population = set(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in seed_set)
    population.update(random_mutation(seed_set, n=population_size))
    population.update(brics_recombination(seed_set, n=population_size // 2))
    population = list(population)[:population_size]
    if not population:
        log_fn("[HATA] Geçerli tohum molekül yok, GA başlatılamadı.")
        return []

    n_elite = max(1, int(len(population) * elite_frac))
    history = []

    for gen in range(1, generations + 1):
        scores = score_population(population, docking_opts)
        # Daha negatif affinity = daha iyi fitness → küçükten büyüğe sırala.
        ranked = sorted(population, key=lambda s: scores.get(s, 999.0))
        best_smi = ranked[0]
        best_score = scores.get(best_smi, 999.0)
        history.append((best_smi, best_score))
        log_fn(f"[Nesil {gen:2d}/{generations}] en iyi: {best_score:>8.3f} kcal/mol  {best_smi}")

        if gen == generations:
            # Son nesli skorlarıyla döndür.
            return [(s, scores.get(s, 999.0)) for s in ranked]

        # --- Yeni nesil: elit + çaprazlama/mutasyon ---
        elites = ranked[:n_elite]
        new_pop = set(elites)
        guard = 0
        while len(new_pop) < population_size and guard < population_size * 40:
            guard += 1
            if random.random() < mutation_rate or len(elites) < 2:
                parent = random.choice(elites)
                child = mutate_once(parent)
            else:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
            if child and is_reasonable(child):
                new_pop.add(child)
        population = list(new_pop)[:population_size]

    # Buraya normalde ulaşılmaz.
    final_scores = score_population(population, docking_opts)
    ranked = sorted(population, key=lambda s: final_scores.get(s, 999.0))
    return [(s, final_scores.get(s, 999.0)) for s in ranked]


# ============================================================================
# d) HAZIR (ÖNCEDEN EĞİTİLMİŞ) MODEL DESTEĞİ — opsiyonel plugin
# ============================================================================
def generate_with_pretrained_model(seeds: list[str], n: int, model_config: dict | None = None) -> list[str]:
    """Opsiyonel plugin arayüzü — REINVENT, MolGPT gibi HAZIR/önceden eğitilmiş
    üretken modeller için.

    Bu fonksiyon KASITLI olarak bir "stub"tır: sistemin çekirdeği bu olmadan tam
    çalışır. Böyle bir modeli entegre etmek istersen:

        1. Modeli ayrıca kur (ör. `pip install reinvent`) ve checkpoint'ini indir.
        2. Aşağıdaki gövdeyi modelin Python API'siyle doldur — `seeds`'i scaffold
           / prior olarak ver, `n` adet örnek iste, dönen SMILES listesini
           canonical_or_none / is_reasonable ile süzüp döndür.
        3. app.py ve CLI'de --method pretrained zaten bu fonksiyonu çağırır.

    Şu an bu bağımlılık yoksa kullanıcıyı bilgilendirir ve boş liste döner (pipeline
    kırılmaz).
    """
    raise NotImplementedError(
        "Hazır model plugin'i (ör. REINVENT) kurulu değil. Bu yöntem opsiyoneldir; "
        "generate_with_pretrained_model() gövdesini modelinin API'siyle doldur. "
        "Diğer yöntemler (random / brics / genetic) bu olmadan tam çalışır."
    )


# ============================================================================
# ÇIKTI / CLI
# ============================================================================
def write_smi(smiles_list, output_path, prefix="gen", scores=None):
    """Üretilen SMILES'leri pipeline'ın beklediği .smi formatında yazar
    ("SMILES  isim"). `scores` verilirse isim yorum satırında skoru da içerir."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SMILES  isim   (molecule_generator.py tarafından üretildi)"]
    for i, smi in enumerate(smiles_list):
        name = f"{prefix}_{i:04d}"
        if scores is not None and smi in scores:
            lines.append(f"{smi}  {name}  # affinity={scores[smi]:.3f}")
        else:
            lines.append(f"{smi}  {name}")
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def _read_seeds(args) -> list[str]:
    """CLI'den tohumları toplar: --seeds (doğrudan SMILES) ve/veya --seeds-file."""
    seeds = list(args.seeds or [])
    if args.seeds_file:
        for line in Path(args.seeds_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(line.split()[0])
    return seeds


def main():
    parser = argparse.ArgumentParser(
        description="Kural tabanlı yeni molekül üretimi (model eğitimi gerektirmez)"
    )
    parser.add_argument("--method", required=True,
                        choices=["random", "brics", "genetic", "pretrained"])
    parser.add_argument("--seeds", nargs="*", help="Tohum SMILES'ler (boşlukla ayrılmış)")
    parser.add_argument("--seeds-file", help="Tohumları içeren .smi dosyası")
    parser.add_argument("--n", type=int, default=50, help="Üretilecek molekül sayısı (random/brics)")
    parser.add_argument("--output", default="data/generated.smi", help="Çıktı .smi dosyası")
    # Genetik algoritma parametreleri
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--elite-frac", type=float, default=0.20)
    parser.add_argument("--mutation-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=None, help="Rastgelelik tohumu (tekrarlanabilirlik)")
    # GA docking (opsiyonel — verilmezse yedek QED fitness kullanılır)
    parser.add_argument("--receptor", help="Reseptör PDBQT (GA gerçek docking için)")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, default=[20, 20, 20], metavar=("SX", "SY", "SZ"))
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--workdir", default="results/ga_work", help="GA docking ara dosyaları")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    seeds = _read_seeds(args)
    if not seeds:
        parser.error("En az bir tohum molekül ver: --seeds veya --seeds-file")

    if args.method == "random":
        mols = random_mutation(seeds, n=args.n)
        print(f"[OK] {len(mols)} yeni molekül üretildi (rastgele mutasyon).")
        write_smi(mols, args.output)

    elif args.method == "brics":
        mols = brics_recombination(seeds, n=args.n)
        print(f"[OK] {len(mols)} yeni molekül üretildi (BRICS rekombinasyonu).")
        write_smi(mols, args.output)

    elif args.method == "genetic":
        docking_opts = None
        if args.receptor and args.center:
            docking_opts = {
                "receptor": args.receptor, "center": args.center,
                "box_size": args.size, "exhaustiveness": args.exhaustiveness,
                "workdir": args.workdir,
            }
            print(f"[INFO] Gerçek Vina docking ile skorlama: {args.receptor}")
        else:
            print("[INFO] Reseptör verilmedi → QED tabanlı yedek fitness kullanılacak.")
        final = genetic_algorithm(
            seeds, generations=args.generations, population_size=args.population,
            elite_frac=args.elite_frac, mutation_rate=args.mutation_rate,
            docking_opts=docking_opts,
        )
        mols = [s for s, _ in final]
        scores = {s: sc for s, sc in final}
        print(f"[OK] GA tamamlandı, {len(mols)} molekül son popülasyonda.")
        write_smi(mols, args.output, scores=scores)

    elif args.method == "pretrained":
        try:
            mols = generate_with_pretrained_model(seeds, n=args.n)
            write_smi(mols, args.output)
        except NotImplementedError as e:
            print(f"[UYARI] {e}")
            return

    print(f"[OK] Çıktı yazıldı: {args.output}")
    print("     Sonraki adım: bu .smi dosyasını ligand_prep → docking → admet → rank zincirine ver.")


if __name__ == "__main__":
    main()
