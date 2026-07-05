# Copyright (C) 2026 Leo
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# See the LICENSE file in the project root for full terms.

"""
known_ligands.py — Belirli bir UniProt ID'ye ait bilinen inhibitörleri
ChEMBL REST API'sinden (fallback: PubChem PUG REST) çeken modül.

Kullanım:
    from known_ligands import fetch_known_ligands
    ligands = fetch_known_ligands("P30405", max_results=5)
    # [{"name": ..., "smiles": ..., "source": "ChEMBL", "activity": "IC50: 23 nM"}, ...]
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 15  # saniye


# ============================================================================
# ChEMBL
# ============================================================================

def _chembl_target_id(uniprot_id: str) -> Optional[str]:
    """UniProt ID'den ChEMBL target ChEMBL ID'sini döndürür."""
    url = (
        f"{CHEMBL_BASE}/target.json"
        f"?target_components__accession={uniprot_id}"
        f"&format=json&limit=1"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        targets = data.get("targets", [])
        if targets:
            return targets[0].get("target_chembl_id")
    except Exception as exc:
        logger.warning("ChEMBL target lookup failed: %s", exc)
    return None


def _chembl_activities(target_chembl_id: str, max_results: int) -> list[dict]:
    """Belirtilen ChEMBL target'a karşı en aktif bileşikleri çeker."""
    url = (
        f"{CHEMBL_BASE}/activity.json"
        f"?target_chembl_id={target_chembl_id}"
        f"&standard_type__in=IC50,Ki,Kd,EC50"
        f"&standard_relation__in==,<"
        f"&standard_value__isnull=false"
        f"&assay_type=B"
        f"&limit={max_results * 5}"  # fazla çek; SMILES olmayanları eleyeceğiz
        f"&order_by=standard_value"
        f"&format=json"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("activities", [])
    except Exception as exc:
        logger.warning("ChEMBL activities fetch failed: %s", exc)
    return []


def _chembl_smiles(molecule_chembl_id: str) -> Optional[str]:
    """ChEMBL molecule ID'den canonical SMILES'i getirir."""
    url = f"{CHEMBL_BASE}/molecule/{molecule_chembl_id}.json?format=json"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        props = r.json().get("molecule_structures") or {}
        return props.get("canonical_smiles")
    except Exception as exc:
        logger.warning("ChEMBL SMILES fetch failed for %s: %s", molecule_chembl_id, exc)
    return None


def _fetch_from_chembl(uniprot_id: str, max_results: int) -> list[dict]:
    """ChEMBL'den bilinen ligandları çek; hata olursa boş liste döndür."""
    target_id = _chembl_target_id(uniprot_id)
    if not target_id:
        logger.info("ChEMBL target not found for %s", uniprot_id)
        return []

    activities = _chembl_activities(target_id, max_results)
    if not activities:
        return []

    seen_smiles: set[str] = set()
    results: list[dict] = []

    for act in activities:
        if len(results) >= max_results:
            break

        mol_id = act.get("molecule_chembl_id")
        if not mol_id:
            continue

        # Aktivite değerini oku
        val = act.get("standard_value")
        unit = act.get("standard_units", "nM")
        act_type = act.get("standard_type", "IC50")
        try:
            val_float = float(val)
            activity_str = f"{act_type}: {val_float:.1f} {unit}"
        except (TypeError, ValueError):
            activity_str = f"{act_type}: bilinmiyor"

        # Molekül adı
        mol_name = act.get("molecule_pref_name") or mol_id

        # SMILES: önce aktivite nesnesinden, yoksa ayrı istek
        smiles = act.get("canonical_smiles")
        if not smiles:
            smiles = _chembl_smiles(mol_id)
        if not smiles:
            continue

        # Duplicate SMILES'i atla
        smi_key = smiles[:40]
        if smi_key in seen_smiles:
            continue
        seen_smiles.add(smi_key)

        results.append({
            "name": mol_name,
            "smiles": smiles,
            "source": "ChEMBL",
            "chembl_id": mol_id,
            "activity": activity_str,
        })

    return results


# ============================================================================
# PubChem fallback
# ============================================================================

def _fetch_from_pubchem(uniprot_id: str, max_results: int) -> list[dict]:
    """
    PubChem PUG REST'ten UniProt ID ile ilişkili bileşikleri çeker.
    ChEMBL kadar aktivite verisi olmaz ama SMILES döndürür.
    """
    # PubChem protein target arama
    url = (
        f"{PUBCHEM_BASE}/assay/target/ProteinGI/{uniprot_id}/aids/JSON"
    )
    # Alternatif yol: doğrudan UniProt ile
    url2 = (
        f"{PUBCHEM_BASE}/assay/target/UniProtID/{uniprot_id}/aids/JSON"
    )

    aids: list[int] = []
    for endpoint in (url2, url):
        try:
            r = requests.get(endpoint, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            aid_list = (
                data.get("InformationList", {}).get("Information", [{}])[0]
                .get("AID", [])
            )
            # Düz liste olabilir
            if isinstance(aid_list, list):
                aids = [int(a) for a in aid_list[:10]]
            elif isinstance(aid_list, int):
                aids = [aid_list]
            if aids:
                break
        except Exception as exc:
            logger.debug("PubChem AID lookup failed (%s): %s", endpoint, exc)

    if not aids:
        return []

    # İlk assay'daki aktif CID'leri çek
    results: list[dict] = []
    seen_cids: set[int] = set()

    for aid in aids[:3]:  # ilk 3 assay yeterli
        if len(results) >= max_results:
            break
        try:
            cid_url = (
                f"{PUBCHEM_BASE}/assay/aid/{aid}/cids/JSON"
                f"?cids_type=active&list_return=listkey"
            )
            r = requests.get(cid_url, timeout=TIMEOUT)
            r.raise_for_status()
            cid_data = r.json()
            # Bazı yanıtlar doğrudan CID listesi döndürür
            inner = cid_data.get("IdentifierList", cid_data.get("InformationList", {}))
            cids = inner.get("CID", [])
            if isinstance(cids, int):
                cids = [cids]
        except Exception as exc:
            logger.debug("PubChem CID fetch failed for AID %s: %s", aid, exc)
            continue

        for cid in cids[:max_results]:
            if len(results) >= max_results:
                break
            if cid in seen_cids:
                continue
            seen_cids.add(cid)

            # SMILES
            try:
                smi_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/IsomericSMILES,IUPACName/JSON"
                rs = requests.get(smi_url, timeout=TIMEOUT)
                rs.raise_for_status()
                props = rs.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                smiles = props.get("IsomericSMILES", "")
                name = props.get("IUPACName", f"CID_{cid}")
                if not smiles:
                    continue
                results.append({
                    "name": name,
                    "smiles": smiles,
                    "source": "PubChem",
                    "chembl_id": f"CID:{cid}",
                    "activity": "Aktif (AID " + str(aid) + ")",
                })
            except Exception as exc:
                logger.debug("PubChem SMILES fetch failed for CID %s: %s", cid, exc)

    return results


# ============================================================================
# Genel arayüz
# ============================================================================

def fetch_known_ligands(
    uniprot_id: str,
    max_results: int = 5,
) -> tuple[list[dict], str]:
    """
    Belirtilen UniProt ID için bilinen inhibitörleri döndürür.

    Returns
    -------
    (ligands, message) :
        ligands : her biri {"name", "smiles", "source", "activity"} dict'i olan liste
        message : kullanıcıya gösterilecek durum mesajı
    """
    if not uniprot_id or not uniprot_id.strip():
        return [], "UniProt ID boş, bilinen ligand aranamadı."

    uid = uniprot_id.strip().upper()

    # 1) ChEMBL dene
    try:
        ligands = _fetch_from_chembl(uid, max_results)
        if ligands:
            return ligands, f"ChEMBL'de {uid} hedefine karşı {len(ligands)} bilinen inhibitör bulundu."
    except Exception as exc:
        logger.error("ChEMBL fetch error: %s", exc)

    # 2) PubChem fallback
    try:
        ligands = _fetch_from_pubchem(uid, max_results)
        if ligands:
            return ligands, f"PubChem'de {uid} hedefine karşı {len(ligands)} bileşik bulundu."
    except Exception as exc:
        logger.error("PubChem fetch error: %s", exc)

    # 3) Her iki kaynak da başarısız
    return [], (
        f"⚠️ {uid} için ChEMBL ve PubChem'de otomatik ligand bulunamadı. "
        "Lütfen tohum SMILES'lerini aşağıdaki kutuya elle girin."
    )
