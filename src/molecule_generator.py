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
import json
import random
import time
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit'in sanitization uyarılarını sustur — geçersiz varyantları zaten
# kendimiz eleyeceğiz, log'u kirletmesinler.
RDLogger.DisableLog("rdApp.*")


# ============================================================================
# ÇALIŞTIRMAYA (RUN) ÖZGÜ LİGAND YOLLARI — TEK KAYNAK
# ============================================================================
# Bir GA/füzyon çalıştırmasının ürettiği tüm ligand dosyalarının yolları
# BURADAN üretilir. Hem docking (ligand hazırlama), hem de otomatik doğrulama
# adımı yolları BU fonksiyondan alır. Böylece yol iki ayrı yerde elle
# birleştirilmez ve gelecekte "hazırlanan klasör" ile "doğrulamanın baktığı
# klasör" birbirinden sapamaz (senkronizasyon bozulmasını kökten önler).
def ligand_workspace(workdir) -> dict:
    """Verilen çalıştırma iş dizini (workdir) için ligand alt-klasörlerini döndürür.

    Returns:
        {
          "prepared":   docking için hazırlanan ham .pdbqt'ler (gen_XXXX.pdbqt),
          "poses":      docklanmış pozlar (gen_XXXX_docked.pdbqt),
          "validation": doğrulama için, ga_final_scores.csv ile AYNI adlarla
                        yeniden hazırlanan .pdbqt'ler,
        }
    """
    workdir = Path(workdir)
    return {
        "prepared": workdir / "prepared",
        "poses": workdir / "poses",
        "validation": workdir / "validation_prepared",
    }


def prepare_validation_ligands(final: list, workdir) -> tuple:
    """Doğrulanacak final adayları, ga_final_scores.csv'deki gen_XXXX adlarıyla
    BİREBİR aynı adları kullanarak ayrı bir doğrulama klasörüne yeniden hazırlar.

    Neden gerekli:
        GA/füzyon her nesilde `prepared/` klasörüne gen_0000, gen_0001... adıyla
        yazar ve bu adlar nesil-içi indeks olduğu için nesiller arasında ÜZERİNE
        yazılır. `ga_final_scores.csv` ise adları FINAL popülasyonun sırasına göre
        verir. Dolayısıyla diskteki gen_0003.pdbqt ile CSV'deki gen_0003 satırı
        FARKLI moleküller olabilir → doğrulama yanlış molekülü docklar veya dosyayı
        bulamaz. Bu fonksiyon final adayları kendi SMILES'lerinden, CSV ile aynı
        indekste yeniden hazırlayarak ad↔dosya eşleşmesini GARANTİ eder.

    Returns:
        (validation_dir, errors)  — errors: {gen_adı: hata_sebebi} (boşsa hepsi OK)
    """
    ws = ligand_workspace(workdir)
    val_dir = ws["validation"]
    val_dir.mkdir(parents=True, exist_ok=True)

    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        import ligand_prep
    except Exception as e:
        # Tüm adaylar için tek ortak sebep — modül yüklenemedi.
        return val_dir, {f"gen_{i:04d}": f"ligand_prep import edilemedi: {e}"
                         for i in range(len(final))}

    errors = {}
    for i, (smi, _sc) in enumerate(final):
        name = f"gen_{i:04d}"
        try:
            sdf = ligand_prep.prepare_ligand(smi, name, val_dir)
            if sdf is None:
                errors[name] = f"3D hazırlama başarısız (geçersiz/gömülemeyen SMILES): {smi}"
                continue
            pdbqt = ligand_prep.convert_to_pdbqt(sdf)
            if pdbqt is None:
                errors[name] = "PDBQT dönüşümü başarısız (meeko kurulu mu?)"
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
    return val_dir, errors

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


def score_population(
    smiles_list: list[str],
    docking_opts: dict | None,
    tick_cb=None,
    n_jobs: int | None = None,
) -> tuple[dict[str, float], str]:
    """Popülasyondaki her molekülü skorlar ve hangi modun kullanıldığını döndürür.

    tick_cb: Bir molekül skorlanınca çağrılan opsiyonel geri-çağrı (ilerleme
             çubuğu için). Argümansız çağrılır.
    n_jobs:  Gerçek docking'te kullanılacak paralel işlem sayısı (None → otomatik).
    """
    if not docking_opts:
        scores = {}
        for smi in smiles_list:
            scores[smi] = _pseudo_affinity(smi)
            if tick_cb:
                tick_cb()
        return scores, "qed_fallback"

    scores, mode = _dock_smiles(
        smiles_list,
        receptor=docking_opts["receptor"],
        center=docking_opts["center"],
        box_size=docking_opts["box_size"],
        workdir=docking_opts["workdir"],
        exhaustiveness=docking_opts.get("exhaustiveness", 8),
        tick_cb=tick_cb,
        n_jobs=n_jobs,
    )
    return {smi: (scores.get(smi) if scores.get(smi) is not None else 999.0)
            for smi in smiles_list}, mode


def _resolve_n_jobs(n_jobs: int | None, n_items: int) -> int:
    """Kullanılacak paralel çekirdek sayısını belirler. Codespaces genelde 2-4
    çekirdek verir; hepsini kullanmak yerine biraz pay bırakırız."""
    import os
    if n_jobs is not None and n_jobs > 0:
        return max(1, min(n_jobs, n_items))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, n_items))


def _dock_one_worker(task):
    """Tek bir SMILES için 3D hazırlama + Vina docking yapan işçi fonksiyon
    (multiprocessing.Pool ile ayrı bir çekirdekte çalışır). (SMILES, skor) döner;
    hata olursa skor None'dur. Her işçi Vina'yı cpu=1 ile çalıştırır ki N işçi
    çekirdekleri aşırı-abone (oversubscribe) etmesin."""
    smi, name, receptor, center, box_size, prepared_dir, poses_dir, exhaustiveness = task
    import sys as _sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in _sys.path:
        _sys.path.insert(0, str(src_dir))
    try:
        import ligand_prep  # noqa: E402
        from vina import Vina  # noqa: E402
    except Exception:
        return smi, None
    try:
        sdf = ligand_prep.prepare_ligand(smi, name, Path(prepared_dir))
        if sdf is None:
            return smi, None
        pdbqt = ligand_prep.convert_to_pdbqt(sdf)
        if pdbqt is None:
            return smi, None
        v = Vina(sf_name="vina", cpu=1, verbosity=0)
        v.set_receptor(str(receptor))
        v.set_ligand_from_file(str(pdbqt))
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=10)
        best_score = float(v.energies(n_poses=1)[0][0])
        try:
            Path(poses_dir).mkdir(parents=True, exist_ok=True)
            v.write_poses(str(Path(poses_dir) / f"{name}_docked.pdbqt"), n_poses=1, overwrite=True)
        except Exception:
            pass
        return smi, best_score
    except Exception:
        return smi, None


def _dock_smiles(smiles_list, receptor, center, box_size, workdir, exhaustiveness=8,
                 tick_cb=None, n_jobs=None):
    """SMILES listesini gerçek Vina docking'inden geçirir ve {SMILES: affinity}
    döndürür. Docking'i multiprocessing.Pool ile birden fazla çekirdekte PARALEL
    çalıştırır (Codespaces'in sınırlı CPU'sunu verimli kullanmak için). Vina/meeko
    kurulu değilse ya da paralel çalışma başarısız olursa yedek QED fitness'e düşer."""
    try:
        import sys
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        import ligand_prep  # noqa: F401,E402  (işçi süreçte de kullanılır)
        import vina  # noqa: F401,E402  (kurulu mu diye kontrol)
    except Exception as e:
        print(f"[UYARI] Docking modülleri yüklenemedi ({e}); yedek fitness kullanılıyor.")
        for smi in smiles_list:
            if tick_cb:
                tick_cb()
        return {smi: _pseudo_affinity(smi) for smi in smiles_list}, "qed_fallback"

    # Yollar TEK KAYNAKTAN (ligand_workspace) — doğrulama adımıyla aynı klasörler.
    ws = ligand_workspace(workdir)
    prepared = ws["prepared"]
    poses = ws["poses"]
    prepared.mkdir(parents=True, exist_ok=True)
    poses.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, smi in enumerate(smiles_list):
        name = f"gen_{i:04d}"
        tasks.append((smi, name, str(receptor), list(center), list(box_size),
                      str(prepared), str(poses), exhaustiveness))

    smi_scores = {}
    n_workers = _resolve_n_jobs(n_jobs, len(tasks))
    try:
        if n_workers > 1 and len(tasks) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
            print(f"[INFO] {len(tasks)} molekül {n_workers} çekirdekte paralel docklaniyor...")
            with ctx.Pool(processes=n_workers) as pool:
                for smi, score in pool.imap_unordered(_dock_one_worker, tasks):
                    smi_scores[smi] = score
                    if tick_cb:
                        tick_cb()
        else:
            for task in tasks:
                smi, score = _dock_one_worker(task)
                smi_scores[smi] = score
                if tick_cb:
                    tick_cb()
    except Exception as e:
        print(f"[UYARI] Paralel docking çalışmadı ({e}); yedek fitness kullanılıyor.")
        for smi in smiles_list:
            smi_scores.setdefault(smi, None)
            if tick_cb:
                tick_cb()
        # En azından skorlanamayanlara QED vekili ver ki GA ilerleyebilsin.
        return {smi: (sc if sc is not None else _pseudo_affinity(smi))
                for smi, sc in ((s, smi_scores.get(s)) for s in smiles_list)}, "qed_fallback"

    return smi_scores, "real_docking"


def genetic_algorithm(
    seeds: list[str],
    generations: int = 10,
    population_size: int = 30,
    elite_frac: float = 0.20,
    mutation_rate: float = 0.30,
    docking_opts: dict | None = None,
    log_fn=print,
    refresh_every: int = 3,
    adaptive_mutation: bool = True,
    scaffold_diversity_threshold: float = 0.3,
    tick_cb=None,
    stage_cb=None,
    n_jobs: int | None = None,
) -> tuple[list[tuple[str, float]], str]:
    # --- Başlangıç popülasyonu ---
    seed_set = [s for s in seeds if Chem.MolFromSmiles(s) is not None]
    population = set(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in seed_set)
    population.update(random_mutation(seed_set, n=population_size))
    population.update(brics_recombination(seed_set, n=population_size // 2))
    population = list(population)[:population_size]
    if not population:
        log_fn("[HATA] Geçerli tohum molekül yok, GA başlatılamadı.")
        return [], "qed_fallback"

    n_elite = max(1, int(len(population) * elite_frac))
    history = []
    current_mode = "qed_fallback"

    for gen in range(1, generations + 1):
        scores, current_mode = score_population(population, docking_opts,
                                                tick_cb=tick_cb, n_jobs=n_jobs)
        # Daha negatif affinity = daha iyi fitness → küçükten büyüğe sırala.
        ranked = sorted(population, key=lambda s: scores.get(s, 999.0))
        best_smi = ranked[0]
        best_score = scores.get(best_smi, 999.0)
        history.append((best_smi, best_score))
        if stage_cb:
            stage_cb(f"Genetik optimizasyon — nesil {gen}/{generations}", best_score)
        
        # Çeşitlilik ölçümü
        scaffolds = set()
        for s in population:
            mol = Chem.MolFromSmiles(s)
            if mol:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(scaffold))
        diversity = len(scaffolds) / population_size

        log_fn(f"[Nesil {gen:2d}/{generations}] en iyi: {best_score:>8.3f} kcal/mol | benzersiz scaffold: {len(scaffolds)}/{population_size} (çeşitlilik: %{diversity*100:.0f})")

        # Adaptif mutasyon
        if adaptive_mutation and len(history) >= 3:
            if history[-1][1] >= history[-3][1] and diversity < scaffold_diversity_threshold:
                mutation_rate = min(1.0, mutation_rate * 1.5)
                log_fn(f"  [UYARI] Yakınsama tespit edildi, mutasyon oranı artırıldı: {mutation_rate:.2f}")

        if gen == generations:
            return [(s, scores.get(s, 999.0)) for s in ranked], current_mode

        # --- Yeni nesil: elit + çaprazlama/mutasyon ---
        elites = ranked[:n_elite]
        new_pop = set(elites)
        
        # Tazelik enjeksiyonu
        if gen % refresh_every == 0 and gen < generations:
            log_fn(f"  [TAZELİK ENJEKSİYONU] Popülasyonun en kötü %20'si yeni rastgele bireylerle değiştiriliyor.")
            num_replace = max(1, int(population_size * 0.2))
            fresh = random_mutation(seed_set, n=num_replace)
            for f in fresh:
                new_pop.add(f)

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

    final_scores, current_mode = score_population(population, docking_opts,
                                                  tick_cb=tick_cb, n_jobs=n_jobs)
    ranked = sorted(population, key=lambda s: final_scores.get(s, 999.0))
    return [(s, final_scores.get(s, 999.0)) for s in ranked], current_mode


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
# e) FÜZYON ÜRETİM MOTORU
# ============================================================================
# Füzyon motorunun VARSAYILAN (demo/test için hızlı) parametreleri.
# Kullanıcı isterse büyütebilir; varsayılan bilinçli olarak KÜÇÜK tutulur ki
# Codespaces'in sınırlı CPU'sunda birkaç dakikada bitsin.
FUSION_DEFAULT_POOL = 40          # keşif havuzu (eskiden ~100)
FUSION_DEFAULT_POPULATION = 15    # GA popülasyonu (eskiden 30)
FUSION_DEFAULT_GENERATIONS = 3    # GA nesil sayısı (eskiden 5)
FUSION_DEFAULT_QED = 0.5          # ön eleme QED eşiği (eskiden 0.3 — çok gevşekti)


def fusion_generation(
    seeds: list[str],
    docking_opts: dict | None,
    log_fn=print,
    discovery_pool: int = FUSION_DEFAULT_POOL,
    population_size: int = FUSION_DEFAULT_POPULATION,
    generations: int = FUSION_DEFAULT_GENERATIONS,
    pre_screen_keep: int | None = None,
    qed_threshold: float = FUSION_DEFAULT_QED,
    n_jobs: int | None = None,
    progress_fn=None,
) -> tuple[list[tuple[str, float]], str]:
    """Dört aşamalı füzyon üretim motoru: Geniş keşif -> Ön Eleme -> Genetik Opt. -> Son Rafinasyon.

    progress_fn: Opsiyonel ilerleme geri-çağrısı. Her çağrıldığında bir durum
                 sözlüğü alır: {stage, done, total, best, ...}. UI ilerleme
                 çubuğu ve "X/Y molekül, tahmini kalan Z dk" için bunu kullanır.
    """
    # pre_screen_keep verilmezse GA popülasyonu kadar molekül tut → docking
    # maliyeti üst sınırdan bağlanır (Sorun 3: gereksiz pahalı docking'i önle).
    if pre_screen_keep is None:
        pre_screen_keep = population_size

    # --- İlerleme durumu ---------------------------------------------------
    # total = GA'nın skorlayacağı toplam molekül-docking birimi (pop * nesil)
    #         + rafinasyon havuzu (nesil sonunda kesinleşir, tahminle başlarız).
    state = {
        "stage": "Başlıyor",
        "done": 0,
        "total": population_size * generations + population_size,
        "best": None,
    }

    def _emit():
        if progress_fn:
            progress_fn(dict(state))

    def tick_cb():
        state["done"] += 1
        _emit()

    def stage_cb(label, best):
        state["stage"] = label
        if best is not None and (state["best"] is None or best < state["best"]):
            state["best"] = best
        _emit()

    log_fn("⚡ FÜZYON ÜRETİM MOTORU BAŞLIYOR ⚡")
    stage_cb("Aşama A: Geniş keşif", None)

    # AŞAMA A — GENİŞ KEŞİF
    log_fn("\n--- AŞAMA A: GENİŞ KEŞİF ---")
    pool = set()
    seed_set = [s for s in seeds if Chem.MolFromSmiles(s) is not None]

    half = max(2, discovery_pool // 2)
    pool.update(random_mutation(seed_set, n=half))
    pool.update(brics_recombination(seed_set, n=half))
    pool_list = list(pool)
    log_fn(f"Keşif havuzu boyutu: {len(pool_list)}")

    # AŞAMA B — ÖN ELEME
    # is_reasonable + QED>=eşik + Lipinski ile filtreler, SONRA QED'e göre
    # sıralayıp en iyi `pre_screen_keep` molekülü tutar. Böylece ön eleme hem
    # gerçekten seçici olur hem de kaç molekülün pahalı docking'e gireceği
    # üst sınırdan bağlanır.
    log_fn("\n--- AŞAMA B: ÖN ELEME ---")
    stage_cb("Aşama B: Ön eleme", state["best"])

    import sys
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import admet_filter

    scored_candidates = []  # (smi, qed)
    for smi in pool_list:
        if not is_reasonable(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        try:
            q = QED.qed(mol)
        except Exception:
            q = 0.0
        if q < qed_threshold:
            continue
        if admet_filter.lipinski_veber_filter(smi, "test").get("pass"):
            scored_candidates.append((smi, q))

    # QED'e göre (yüksekten düşüğe) sırala ve en iyileri tut.
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    passed_pre_screen = [smi for smi, _ in scored_candidates[:pre_screen_keep]]

    log_fn(f"Ön elemeyi geçen (Makul + QED>={qed_threshold} + Lipinski): "
           f"{len(scored_candidates)}/{len(pool_list)} · "
           f"docking'e giren (en iyi {pre_screen_keep}): {len(passed_pre_screen)}")
    if not passed_pre_screen:
        log_fn("Ön elemeyi geçen molekül kalmadı! Tohumlarla devam ediliyor.")
        passed_pre_screen = seed_set

    # AŞAMA C — GENETİK OPTİMİZASYON
    log_fn("\n--- AŞAMA C: GENETİK OPTİMİZASYON ---")
    ga_results, mode = genetic_algorithm(
        passed_pre_screen,
        generations=generations,
        population_size=population_size,
        docking_opts=docking_opts,
        log_fn=log_fn,
        tick_cb=tick_cb,
        stage_cb=stage_cb,
        n_jobs=n_jobs,
    )
    if not ga_results:
        return [], "qed_fallback"

    top_ga = [smi for smi, sc in ga_results[:5]]

    # AŞAMA D — SON RAFİNASYON
    log_fn("\n--- AŞAMA D: SON RAFİNASYON ---")
    refined_pool = set(top_ga)
    for smi in top_ga:
        for _ in range(3):
            mut = mutate_once(smi)
            if mut and is_reasonable(mut):
                refined_pool.add(mut)

    refined_list = list(refined_pool)
    log_fn(f"Rafinasyon havuzu (Top GA + küçük mutasyonlar): {len(refined_list)}")
    # total'ı rafinasyon havuzunun gerçek boyutuna göre düzelt (ETA doğrulansın).
    state["total"] = state["done"] + len(refined_list)
    stage_cb("Aşama D: Son rafinasyon", state["best"])

    final_scores, final_mode = score_population(refined_list, docking_opts,
                                                tick_cb=tick_cb, n_jobs=n_jobs)
    final_ranked = sorted(refined_list, key=lambda s: final_scores.get(s, 999.0))
    final_output = [(s, final_scores.get(s, 999.0)) for s in final_ranked[:5]]
    if final_output:
        stage_cb("Tamamlandı", final_output[0][1])

    log_fn("\n--- FÜZYON ÖZETİ ---")
    log_fn(f"Keşif: {len(pool_list)} → Ön eleme: {len(passed_pre_screen)} → "
           f"GA ({generations} nesil) → Rafinasyon: {len(refined_list)} → "
           f"Final: {len(final_output)} aday")
    return final_output, final_mode


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


class _ProgressWriter:
    """Füzyon ilerlemesini atomik olarak bir JSON dosyasına yazar. Streamlit UI
    bu dosyayı periyodik okuyarak ilerleme çubuğunu SÜRECİ YENİDEN BAŞLATMADAN
    günceller (Sorun 1). Yazma throttle'lıdır (aşırı disk I/O olmasın)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()
        self.log: list[str] = []
        self.state = {"status": "running", "stage": "Başlıyor",
                      "done": 0, "total": 0, "best": None, "started_at": self.start}
        self._last_flush = 0.0
        self._flush(force=True)

    def log_fn(self, msg):
        print(msg, flush=True)
        for ln in str(msg).splitlines():
            if ln.strip():
                self.log.append(ln)
        self._flush()

    def progress_fn(self, st):
        self.state.update(st)
        self._flush()

    def finish(self, results, mode, validation=None):
        self.state["status"] = "done"
        self.state["mode"] = mode
        self.state["results"] = [[s, sc] for s, sc in results]
        if validation is not None:
            # Doğrulama özeti/hatası — UI 'done' aşamasında bunu gösterir.
            self.state["validation"] = validation
        self._flush(force=True)

    def error(self, msg):
        self.state["status"] = "error"
        self.state["error"] = str(msg)
        self._flush(force=True)

    def _flush(self, force=False):
        now = time.time()
        if not force and (now - self._last_flush) < 0.3:
            return
        self._last_flush = now
        data = dict(self.state)
        data["log"] = self.log[-15:]
        data["elapsed"] = now - self.start
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(data))
            tmp.replace(self.path)
        except Exception:
            pass


def _run_auto_validation(args, final: list, mode: str) -> str:
    """GA/füzyon bittiğinde en iyi adayları yüksek exhaustiveness ile yeniden
    docklayarak doğrular. Ligand yollarını `ligand_workspace`'ten (TEK KAYNAK)
    alır ve final adayları CSV ile aynı adlarla yeniden hazırlar (ad↔dosya
    eşleşmesi garanti). UI'da gösterilecek kısa bir özet metni döndürür.

    Bu fonksiyon HİÇBİR koşulda exception fırlatmaz — her durumda anlamlı bir
    özet döndürür, çünkü çağıran taraf (füzyon) bunu progress.json'a yazar.
    """
    import csv as _csv
    import traceback as _tb

    ws = ligand_workspace(args.workdir)

    try:
        # 1) GA skorlarını validate_top_candidates'in beklediği formata yaz.
        ga_scores_csv = Path(args.workdir) / "ga_final_scores.csv"
        ga_scores_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(ga_scores_csv, "w", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=["ligand", "SMILES", "affinity_kcal_mol", "skor_kaynagi"])
            _w.writeheader()
            for i, (smi, sc) in enumerate(final):
                _w.writerow({"ligand": f"gen_{i:04d}", "SMILES": smi,
                             "affinity_kcal_mol": sc, "skor_kaynagi": mode})

        # 2) Final adayları CSV ile AYNI adlarla yeniden hazırla (ad↔dosya eşleşmesi).
        val_dir, prep_errors = prepare_validation_ligands(final, args.workdir)

        # 3) Doğrulamayı çalıştır. Ligand dizinleri TEK KAYNAKTAN gelir; önce
        #    ada birebir uyan validation klasörü, sonra ham prepared/poses.
        import validate_top_candidates as _vtc
        val_exhaustiveness = args.exhaustiveness * 4  # ör. 8 → 32
        val_output = Path("results/validated_candidates.csv")
        print(f"\n[DOĞRULAMA] En iyi 5 aday exhaustiveness={val_exhaustiveness} ile yeniden docklaniyor...")
        val_rows = _vtc.validate_top_candidates(
            scores_csv=ga_scores_csv,
            receptor_pdbqt=Path(args.receptor),
            center=list(args.center),
            box_size=list(args.size),
            top_n=5,
            exhaustiveness=val_exhaustiveness,
            ligand_dirs=[val_dir, ws["prepared"], ws["poses"]],
            output_csv=val_output,
            prep_errors=prep_errors,
        )
    except Exception as _ve:
        # Beklenmedik hata — özeti üret, gene de bir CSV yazmayı dene.
        err = f"{type(_ve).__name__}: {_ve}"
        print(f"[UYARI] Otomatik doğrulama çalışmadı: {err}")
        print(_tb.format_exc())
        try:
            _vtc = __import__("validate_top_candidates")
            _vtc.write_failure_csv(
                Path("results/validated_candidates.csv"),
                [(f"gen_{i:04d}", sc) for i, (smi, sc) in enumerate(final[:5])],
                sebep=err,
            )
        except Exception:
            pass
        return f"⚠️ Doğrulama çalışmadı — sebep: {err}"

    # 4) Konsol özeti + UI özet metni üret.
    if not val_rows:
        return "⚠️ Doğrulama: skorlanacak aday bulunamadı (validated_candidates.csv yine de yazıldı)."

    ok = sum(1 for vr in val_rows if str(vr.get("guven_durumu", "")).startswith(("GÜVENİLİR", "GÜÇLÜ", "ŞÜPHELİ", "ARTEFAKT")))
    fail = [vr for vr in val_rows if str(vr.get("guven_durumu", "")).startswith("DOĞRULANAMADI")]

    print("\n┌─ DOĞRULAMA ÖZETİ ─────────────────────────────────────────┐")
    print(f"│ {'Ligand':<14} {'İlk Skor':>9} {'Doğrulama':>10} {'Fark':>6} {'Durum':<30} │")
    print("│ " + "─" * 74 + " │")
    for vr in val_rows:
        ilk = vr.get('ilk_skor', '')
        dog = vr.get('dogrulanmis_skor', '')
        frk = vr.get('fark', '')
        dur = vr.get('guven_durumu', '')
        ilk_s = f"{ilk:.3f}" if isinstance(ilk, (int, float)) else str(ilk)
        dog_s = f"{dog:.3f}" if isinstance(dog, (int, float)) else str(dog)
        frk_s = f"{frk:.2f}" if isinstance(frk, (int, float)) else str(frk)
        sembol = {"GÜVENİLİR": "✓", "ŞÜPHELİ — tekrar kontrol et": "⚠", "ARTEFAKT OLASI — güvenme": "✗", "GÜÇLÜ ADAY — ilk tarama hafife almış": "⭐"}.get(dur, "?")
        print(f"│ {vr['ligand']:<14} {ilk_s:>9} {dog_s:>10} {frk_s:>6} {sembol} {dur:<28} │")
    print("└" + "─" * 76 + "┘")
    print("[OK] Detaylı doğrulama raporu: results/validated_candidates.csv")

    if fail:
        # İlk başarısızlığın GERÇEK sebebini özete taşı — sessizce yutma.
        ilk_sebep = str(fail[0].get("sebep", "") or fail[0].get("guven_durumu", "")).strip()
        return (f"⚠️ Doğrulama: {ok}/{len(val_rows)} aday doğrulandı, "
                f"{len(fail)} başarısız — sebep: {ilk_sebep}")
    return f"✅ Doğrulama tamamlandı: {ok}/{len(val_rows)} aday doğrulandı."


def main():
    parser = argparse.ArgumentParser(
        description="Kural tabanlı yeni molekül üretimi (model eğitimi gerektirmez)"
    )
    parser.add_argument("--method", required=True,
                        choices=["random", "brics", "genetic", "pretrained", "fusion"])
    parser.add_argument("--seeds", nargs="*", help="Tohum SMILES'ler (boşlukla ayrılmış)")
    parser.add_argument("--seeds-file", help="Tohumları içeren .smi dosyası")
    parser.add_argument("--n", type=int, default=50, help="Üretilecek molekül sayısı (random/brics)")
    parser.add_argument("--output", default="data/generated.smi", help="Çıktı .smi dosyası")
    # Genetik algoritma parametreleri
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--elite-frac", type=float, default=0.20)
    parser.add_argument("--mutation-rate", type=float, default=0.30)
    parser.add_argument("--refresh-every", type=int, default=3, help="Kaç nesilde bir tazelik enjeksiyonu yapılsın")
    parser.add_argument("--adaptive-mutation", action="store_true", default=True, help="Adaptif mutasyonu aç")
    parser.add_argument("--scaffold-diversity-threshold", type=float, default=0.3, help="Çeşitlilik sınırı")
    parser.add_argument("--seed", type=int, default=None, help="Rastgelelik tohumu (tekrarlanabilirlik)")
    # GA docking (opsiyonel — verilmezse yedek QED fitness kullanılır)
    parser.add_argument("--receptor", help="Reseptör PDBQT (GA gerçek docking için)")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, default=[20, 20, 20], metavar=("SX", "SY", "SZ"))
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--workdir", default="results/ga_work", help="GA docking ara dosyaları")
    # Füzyon motoru parametreleri (VARSAYILAN küçük/hızlı — kullanıcı büyütebilir)
    parser.add_argument("--fusion-pool", type=int, default=FUSION_DEFAULT_POOL,
                        help="Füzyon keşif havuzu boyutu (varsayılan hızlı: küçük)")
    parser.add_argument("--fusion-population", type=int, default=FUSION_DEFAULT_POPULATION,
                        help="Füzyon GA popülasyon boyutu")
    parser.add_argument("--fusion-generations", type=int, default=FUSION_DEFAULT_GENERATIONS,
                        help="Füzyon GA nesil sayısı")
    parser.add_argument("--fusion-qed", type=float, default=FUSION_DEFAULT_QED,
                        help="Füzyon ön eleme QED eşiği")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Docking için paralel çekirdek sayısı (varsayılan: otomatik)")
    parser.add_argument("--progress-file", default=None,
                        help="İlerleme durumunun JSON olarak yazılacağı dosya (UI için)")
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

    elif args.method in ("genetic", "fusion"):
        writer = None  # füzyon ilerleme yazıcısı (varsa) — doğrulamadan SONRA finish edilir
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
            
        if args.method == "genetic":
            final, mode = genetic_algorithm(
                seeds, generations=args.generations, population_size=args.population,
                elite_frac=args.elite_frac, mutation_rate=args.mutation_rate,
                docking_opts=docking_opts,
                refresh_every=args.refresh_every,
                adaptive_mutation=args.adaptive_mutation,
                scaffold_diversity_threshold=args.scaffold_diversity_threshold,
                n_jobs=args.n_jobs,
            )
        else:
            # Füzyon: opsiyonel ilerleme dosyası (UI subprocess ile çağırınca kullanılır)
            writer = _ProgressWriter(args.progress_file) if args.progress_file else None
            fusion_log = writer.log_fn if writer else print
            fusion_progress = writer.progress_fn if writer else None
            try:
                final, mode = fusion_generation(
                    seeds, docking_opts=docking_opts,
                    log_fn=fusion_log,
                    discovery_pool=args.fusion_pool,
                    population_size=args.fusion_population,
                    generations=args.fusion_generations,
                    qed_threshold=args.fusion_qed,
                    n_jobs=args.n_jobs,
                    progress_fn=fusion_progress,
                )
                # NOT: writer.finish() burada DEĞİL — doğrulama adımından SONRA
                # çağrılır ki doğrulama özeti/hatası da progress.json'a yazılıp
                # UI'da gösterilebilsin (aksi halde UI 'done' görüp okumayı bırakır).
            except Exception as e:
                if writer:
                    writer.error(e)
                raise

        if mode == "qed_fallback":
            print("\n┌─────────────────────────────────────────────────────────────┐")
            print("│ ⚠️  UYARI: Reseptör verilmedi. Skorlar GERÇEK DOCKING        │")
            print("│     DEĞİL — sadece QED (ilaç-benzerlik) tahminidir.          │")
            print("│     Gerçek affinity için --receptor ve --center belirt.      │")
            print("└─────────────────────────────────────────────────────────────┘\n")

        mols = [s for s, _ in final]
        scores = {s: sc for s, sc in final}
        print(f"[OK] {args.method.upper()} tamamlandı, {len(mols)} molekül son popülasyonda.")
        write_smi(mols, args.output, scores=scores)

        # ----------------------------------------------------------------
        # OTOMATİK DOĞRULAMA: GA bittiğinde en iyi N adayı yüksek
        # exhaustiveness ile yeniden dockla — artefakt skorları tespit et.
        # ----------------------------------------------------------------
        validation_summary = None
        if args.receptor and args.center:
            validation_summary = _run_auto_validation(args, final, mode)

        # Füzyon: TÜM iş (üretim + doğrulama) bittikten sonra 'done' yaz — böylece
        # doğrulama özeti/hatası da UI'ya ulaşır.
        if writer:
            writer.finish(final, mode, validation=validation_summary)

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
