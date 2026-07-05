import subprocess
import csv
from pathlib import Path
import os

def dock_with_smina(receptor_pdbqt: Path, ligand_pdbqt: Path, center: list[float], box_size: list[float], exhaustiveness: int = 8) -> float | None:
    smina_cmd = [
        "smina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--log", os.devnull
    ]
    try:
        result = subprocess.run(smina_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if result.returncode == 127 or "command not found" in result.stderr.lower():
                print("Smina kurulu değil, kurulum: conda install -c conda-forge smina")
            return None
        
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    except FileNotFoundError:
        print("Smina kurulu değil, kurulum: conda install -c conda-forge smina")
        return None
    return None

def cross_validate(smiles_csv: Path, receptor_pdbqt: Path, center: list[float], box_size: list[float], ligand_dirs: list[Path], top_n: int = 5):
    # This will read the validated_candidates.csv which has the vina_affinity (dogrulanmis_skor) 
    # But wait, the prompt says smiles_csv. It might be docking_scores.csv or validated_candidates.csv
    # We can just read the top N from the input CSV (like validate_top_candidates does)
    # Actually, we can reuse `load_scores` and `find_ligand_pdbqt` from validate_top_candidates
    import sys
    _SRC_DIR = Path(__file__).resolve().parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    
    import validate_top_candidates as vtc
    all_scores = vtc.load_scores(smiles_csv)
    if not all_scores:
        return []
        
    # We assume scores are already sorted or we sort them
    all_scores.sort(key=lambda r: r["affinity_kcal_mol"])
    top_rows = all_scores[:top_n]
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION — Smina vs Vina")
    print(f"{'='*60}")
    
    for row in top_rows:
        ligand_name = row["ligand"]
        vina_affinity = row.get("dogrulanmis_skor", row["affinity_kcal_mol"]) # If it has dogrulanmis_skor, use it
        
        lig_pdbqt = vtc.find_ligand_pdbqt(ligand_name, ligand_dirs)
        if lig_pdbqt is None:
            results.append({
                "ligand": ligand_name,
                "vina_affinity": vina_affinity,
                "smina_affinity": None,
                "fark": None,
                "tutarlilik_durumu": "Smina testi yapılamadı (PDBQT yok)"
            })
            continue
            
        print(f"  ↻  {ligand_name} Smina ile docklanıyor...", end=" ", flush=True)
        smina_affinity = dock_with_smina(receptor_pdbqt, lig_pdbqt, center, box_size, exhaustiveness=8)
        
        if smina_affinity is None:
            print("BAŞARISIZ (Smina kurulu değil veya hata)")
            results.append({
                "ligand": ligand_name,
                "vina_affinity": vina_affinity,
                "smina_affinity": None,
                "fark": None,
                "tutarlilik_durumu": "Smina testi başarısız"
            })
            continue
            
        fark = abs(vina_affinity - smina_affinity)
        if fark <= 1.5:
            durum = "TUTARLI — iki motor da aynı fikirde"
        else:
            durum = "TUTARSIZ — motorlar arasında anlaşmazlık, dikkatli ol"
            
        print(f"{smina_affinity:.3f} (Δ={fark:.2f}) -> {durum}")
        
        results.append({
            "ligand": ligand_name,
            "vina_affinity": round(vina_affinity, 4),
            "smina_affinity": round(smina_affinity, 4),
            "fark": round(fark, 4),
            "tutarlilik_durumu": durum
        })
        
    return results
