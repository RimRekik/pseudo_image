#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tri des fichiers .wiff / .wiff2 / .scan (et fichiers associés) par classe d'espèce.

Principe :
1. Le fichier Excel de correspondance (colonnes 'sample_name' + 'species') sert de
   référentiel : sample_name = Code_espèce (ex: ESCCOL) + Sample_nb (ex: 217).
2. Chaque fichier .wiff/.wiff2/.scan trouvé dans l'arborescence source est nommé :
       Code_espèce - Sample_nb - milieu - protocole . ext
   On extrait Code_espèce et Sample_nb depuis le nom de fichier, on corrige les
   typos connus (ex: COLI -> ESCCOL), puis on cherche Code_espèce+Sample_nb dans
   le référentiel pour obtenir l'espèce exacte.
3. Si l'espèce appartient à une des 5 classes cibles, le(s) fichier(s) sont copiés
   (jamais déplacés/modifiés) dans DEST_DIR/<Nom de la classe>/.
4. Un rapport CSV détaillé est produit : fichiers classés, ignorés (autre espèce),
   et non résolus (à vérifier manuellement) -- rien n'est classé "au hasard".

Usage :
    python3 sort_by_species.py \
        --source "/chemin/vers/ton_dossier_source" \
        --xlsx   "/chemin/vers/230804_strain_peptides_antibiogram_Enterobacterales_3_.xlsx" \
        [--dest  "/chemin/vers/ms1_processed"]   (par défaut: ./ms1_processed)
        [--move] [--dry-run]

Résultat créé :
    ms1_processed/
    ├── Escherichia coli/
    ├── Klebsiella pneumoniae/
    ├── Enterobacter hormaechei/
    ├── Proteus mirabilis/
    ├── Citrobacter freundii/
    ├── samples_5_classes.csv   (liste des fichiers classés, avec leur destination)
    └── a_verifier.csv          (codes inconnus non génériques, à vérifier manuellement)

Par défaut les fichiers sont COPIÉS (l'original n'est jamais supprimé). Utiliser
--move si vous voulez déplacer au lieu de copier. --dry-run simule sans rien écrire.
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict

try:
    import openpyxl
except ImportError:
    sys.exit("Le module openpyxl est requis : pip install openpyxl")

# ---------------------------------------------------------------------------
# 1) Les 5 classes cibles.
#    Clé = nom de dossier créé dans DEST_DIR.
#    Valeur = fonction qui teste si un nom d'espèce complet (colonne 'species'
#    du xlsx) appartient à cette classe. On matche par préfixe pour absorber
#    les sous-espèces (ex: "Enterobacter hormaechei hoffmannii" -> hormaechei).
# ---------------------------------------------------------------------------
TARGET_CLASSES = {
    "Escherichia coli":         lambda sp: sp.startswith("Escherichia coli"),
    "Klebsiella pneumoniae":    lambda sp: sp.startswith("Klebsiella pneumoniae"),
    "Enterobacter hormaechei":  lambda sp: sp.startswith("Enterobacter hormaechei"),
    "Proteus mirabilis":        lambda sp: sp.startswith("Proteus mirabilis"),
    "Citrobacter freundii":     lambda sp: sp.startswith("Citrobacter freundii"),
}

# ---------------------------------------------------------------------------
# 2) Corrections de typo connues sur le Code_espèce, d'après la doc fournie.
#    Clé = code tel qu'il peut apparaître (en MAJUSCULES) dans un nom de fichier,
#    Valeur = code correct tel qu'il apparaît dans sample_name du xlsx.
#    Ajoutez ici toute autre typo que vous repérez dans le rapport "unresolved".
# ---------------------------------------------------------------------------
KNOWN_TYPO_FIXES = {
    "COLI": "ESCCOL",
    "EC": "ESCCOL",
    "ENTABS": "ENTASB",  # 'ENT-ABS...' -> lettres inversées, vrai code = ENTASB (E. asburiae)
    # "AUTRE_TYPO": "CODE_CORRECT",
}

# Extensions gérées, des plus longues/spécifiques aux plus courtes, pour bien
# séparer le "nom de base" (basename) de l'extension lors du regroupement des
# fichiers compagnons (.wiff + .wiff.scan + .wiff2 partageant le même sample).
# Ajout de .npy pour les données déjà converties/pré-traitées en tableaux numpy.
KNOWN_EXTENSIONS = [".wiff.scan", ".wiff2", ".wiff", ".scan", ".npy"]


def strip_known_extension(filename: str):
    """Retourne (basename_sans_ext, extension) en utilisant KNOWN_EXTENSIONS."""
    for ext in KNOWN_EXTENSIONS:
        if filename.lower().endswith(ext):
            return filename[: -len(ext)], filename[-len(ext):]
    return None, None  # extension non reconnue -> fichier ignoré


def parse_filename(basename: str, known_codes: set, typo_fixes: dict):
    """
    basename : nom de fichier SANS extension.
    Gère 3 variantes de nomenclature rencontrées dans les données réelles :
      1. 'ESCCOL-217-AER_100vW_100SPD'   (cas standard : CODE-NB-reste)
      2. 'CIT-FRE-18-AER-d200'           (code coupé en 2 segments : CIT + FRE)
      3. 'EC 107 AER'                     (séparateur espace au lieu de tiret)
    Retourne (code_brut_affiche, code_corrige, sample_nb, reste) ou
    (None, None, None, None) si aucun format ne matche.
    """
    # Unifie les séparateurs : un espace entre les 3 premiers "mots" compte
    # comme un tiret (cas 'EC 107 AER').
    normalized = re.sub(r"\s+", "-", basename.strip())
    parts = normalized.split("-")
    if len(parts) < 2:
        return None, None, None, None

    # --- Tentative 1 : format standard CODE-NB-reste ---
    code1 = parts[0].strip().upper()
    if re.match(r"^\d+", parts[1].strip()):
        sample_nb = re.match(r"^\d+", parts[1].strip()).group(0)
        reste = "-".join(parts[2:])
        code1_fixed = typo_fixes.get(code1, code1)
        return code1, code1_fixed, sample_nb, reste

    # --- Tentative 2 : code coupé en 2 segments, ex CIT-FRE-18-... ---
    if len(parts) >= 3 and parts[0].isalpha() and parts[1].isalpha():
        merged = (parts[0] + parts[1]).strip().upper()
        merged_fixed = typo_fixes.get(merged, merged)
        m3 = re.match(r"^\d+", parts[2].strip())
        if m3 and merged_fixed in known_codes:
            sample_nb = m3.group(0)
            reste = "-".join(parts[3:])
            return f"{parts[0]}-{parts[1]}", merged_fixed, sample_nb, reste

    return None, None, None, None


def load_reference(xlsx_path: Path):
    """
    Lit le xlsx et construit :
      - ref_by_sample_name : { 'ESCCOL217': {'species': ..., 'sample_name':..., ...} }
      - codes_by_nb        : { '217': {'ESCCOL', ...} }  (pour diagnostic des non-résolus)
      - known_codes        : set of all species codes seen (ex 'ESCCOL','KLEPNE',...)
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Sheet 1"] if "Sheet 1" in wb.sheetnames else wb[wb.sheetnames[0]]

    header_row = None
    for r in range(1, 4):
        vals = [c.value for c in ws[r]]
        if vals and "sample_name" in vals:
            header_row = r
            header = vals
            break
    if header_row is None:
        sys.exit("Impossible de trouver la ligne d'en-tête contenant 'sample_name' dans le xlsx.")

    idx_sample = header.index("sample_name")
    idx_species = header.index("species")
    idx_provider = header.index("provider") if "provider" in header else None
    idx_isolate = header.index("microbial_isolate") if "microbial_isolate" in header else None

    ref_by_sample_name = {}
    codes_by_nb = defaultdict(set)
    known_codes = set()

    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        sample_name = row[idx_sample]
        species = row[idx_species]
        if not sample_name or not species:
            continue
        sample_name = str(sample_name).strip()
        m = re.match(r"^([A-Za-z]+)(\d+)$", sample_name)
        if not m:
            continue
        code, nb = m.group(1).upper(), m.group(2)
        known_codes.add(code)
        codes_by_nb[nb].add(code)
        ref_by_sample_name[f"{code}{nb}"] = {
            "species": species,
            "sample_name": sample_name,
            "provider": row[idx_provider] if idx_provider is not None else None,
            "microbial_isolate": row[idx_isolate] if idx_isolate is not None else None,
        }

    return ref_by_sample_name, codes_by_nb, known_codes


def build_code_to_class(ref_by_sample_name: dict, known_codes: set):
    """
    Construit { code_espece : nom_de_classe } pour tout code dont TOUTES les
    occurrences observées dans le référentiel appartiennent à une seule des 5
    classes cibles (ex: ESCCOL -> toujours Escherichia coli, ENTHOR -> toujours
    une sous-espèce de Enterobacter hormaechei). Permet de classer un fichier
    par le seul code, même si son sample_nb exact n'est pas dans le référentiel
    -- mais UNIQUEMENT quand ce code n'est jamais associé à une autre espèce.
    """
    classes_seen_by_code = defaultdict(set)
    for entry in ref_by_sample_name.values():
        m = re.match(r"^([A-Za-z]+)\d+$", entry["sample_name"])
        if not m:
            continue
        code = m.group(1).upper()
        cls = classify_species(entry["species"])
        classes_seen_by_code[code].add(cls if cls else f"__other__:{entry['species']}")

    code_to_class = {}
    for code, classes in classes_seen_by_code.items():
        if len(classes) == 1:
            only = next(iter(classes))
            if not only.startswith("__other__"):
                code_to_class[code] = only
    return code_to_class


def is_generic_prefix(code: str, known_codes: set) -> bool:
    """
    Retourne True si `code` est un préfixe générique/tronqué d'un ou plusieurs
    codes connus (ex: 'PRT' est un préfixe de PRTMIR, PRTVUL, PRTCOL, PRTPEN,
    PRTTER -> générique et donc AMBIGU). Ces codes ne sont jamais devinés :
    on les exclut systématiquement, même quand un seul code correspondrait,
    conformément à la consigne (ex: 'PRT' -> toujours exclu, jamais assimilé
    à PRTMIR).
    """
    if code in known_codes:
        return False  # code déjà complet et exact, pas générique
    return any(kc.startswith(code) for kc in known_codes)


def classify_species(species_full: str):
    """Retourne le nom de la classe cible ou None."""
    for class_name, test in TARGET_CLASSES.items():
        if test(species_full):
            return class_name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Dossier racine 'filtred_usefull' à parcourir")
    ap.add_argument("--xlsx", required=True, help="Fichier Excel de correspondance souches/espèces")
    ap.add_argument("--dest", default="ms1_processed",
                     help="Dossier de sortie (par défaut: ./ms1_processed), créé avec 1 sous-dossier par classe")
    ap.add_argument("--move", action="store_true", help="Déplacer les fichiers au lieu de les copier")
    ap.add_argument("--dry-run", action="store_true", help="Ne rien écrire, juste simuler et produire le rapport")
    args = ap.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    xlsx_path = Path(args.xlsx)

    if not source.is_dir():
        sys.exit(f"Dossier source introuvable : {source}")
    if not xlsx_path.is_file():
        sys.exit(f"Fichier xlsx introuvable : {xlsx_path}")

    print("Lecture du référentiel Excel ...")
    ref_by_sample_name, codes_by_nb, known_codes = load_reference(xlsx_path)
    code_to_class = build_code_to_class(ref_by_sample_name, known_codes)
    print(f"  -> {len(ref_by_sample_name)} souches référencées, {len(known_codes)} codes espèce connus.")
    print(f"  -> {len(code_to_class)} codes non-ambigus mappés directement à une classe cible : {sorted(code_to_class)}")

    if not args.dry_run:
        for class_name in TARGET_CLASSES:
            (dest / class_name).mkdir(parents=True, exist_ok=True)

    # 1) Regrouper tous les fichiers gérés par "basename" (sans extension), pour
    #    copier ensemble .wiff / .wiff2 / .scan d'un même sample.
    groups = defaultdict(list)  # basename -> [Path, ...]
    for p in source.rglob("*"):
        if not p.is_file():
            continue
        base, ext = strip_known_extension(p.name)
        if base is None:
            continue  # extension non gérée (.mzML, .txt, etc.) -> ignoré silencieusement
        groups[(p.parent, base)].append(p)

    print(f"  -> {len(groups)} fichiers/samples .wiff/.wiff2/.scan trouvés dans l'arborescence.")

    report_rows = []
    n_classified = 0
    n_ignored_other_species = 0
    n_unresolved = 0

    for (parent, base), files in sorted(groups.items()):
        code_raw, code_corrected, sample_nb, reste = parse_filename(base, known_codes, KNOWN_TYPO_FIXES)
        status = ""
        species_found = ""
        class_assigned = ""
        code_used = ""

        if code_raw is None:
            status = "FORMAT_NON_RECONNU"
            n_unresolved += 1
        else:
            code_used = code_corrected
            key = f"{code_corrected}{sample_nb}"
            entry = ref_by_sample_name.get(key)

            if entry is None:
                if is_generic_prefix(code_corrected, known_codes):
                    # Code générique/tronqué (ex: 'PRT', 'KLE', 'SER'...) qui
                    # correspond au genre mais pas à une espèce précise.
                    # On l'exclut TOUJOURS, sans jamais deviner l'espèce,
                    # même si un seul candidat existait pour ce sample_nb.
                    status = "EXCLU_CODE_GENERIQUE_AMBIGU"
                    n_ignored_other_species += 1
                elif code_corrected in code_to_class:
                    # Le sample_nb exact n'est pas dans le référentiel, MAIS le
                    # code lui-même correspond sans ambiguïté à une des 5 classes
                    # cibles (jamais observé associé à une autre espèce dans le
                    # référentiel) -> on classe quand même, en le signalant.
                    class_assigned = code_to_class[code_corrected]
                    status = "CLASSE (sample_nb absent du excel, classe par code seul)"
                    n_classified += 1
                elif code_corrected in known_codes:
                    # Code complet et connu, mais qui correspond à une espèce
                    # HORS des 5 classes cibles -> on l'ignore.
                    status = "IGNORE_AUTRE_ESPECE (sample_nb absent du excel)"
                    n_ignored_other_species += 1
                else:
                    # Code inconnu du référentiel et pas un préfixe générique
                    # reconnu : probablement une typo non encore répertoriée.
                    # On ne devine pas non plus -> à vérifier manuellement.
                    candidates = codes_by_nb.get(sample_nb, set())
                    if len(candidates) == 1:
                        status = f"NON_RESOLU_CODE_INCONNU (suggestion: {next(iter(candidates))}{sample_nb})"
                    else:
                        status = "NON_RESOLU_CODE_INCONNU"
                    n_unresolved += 1
            else:
                species_found = entry["species"]
                class_assigned = classify_species(species_found)
                if class_assigned:
                    status = "CLASSE" if code_raw == code_corrected else f"CLASSE (typo corrigee: {code_raw}->{code_corrected})"
                    n_classified += 1
                else:
                    status = "IGNORE_AUTRE_ESPECE"
                    n_ignored_other_species += 1

        # Exécution : copie/déplacement effectif si une classe a été assignée
        for f in files:
            dest_path = ""
            if class_assigned and not args.dry_run:
                target_dir = dest / class_assigned
                target_path = target_dir / f.name
                if args.move:
                    shutil.move(str(f), str(target_path))
                else:
                    shutil.copy2(str(f), str(target_path))
                dest_path = str(target_path)
            elif class_assigned and args.dry_run:
                dest_path = str(dest / class_assigned / f.name)

            report_rows.append({
                "fichier_source": str(f),
                "basename": base,
                "code_brut": code_raw or "",
                "code_utilise": code_used,
                "sample_nb": sample_nb or "",
                "espece_trouvee": species_found,
                "classe_assignee": class_assigned or "",
                "statut": status,
                "destination": dest_path,
            })

    # --- Rapports CSV : on ne garde le détail que pour ce qui est utile ---
    suffix = "_dryrun" if args.dry_run else ""
    out_dir = dest if not args.dry_run else Path(".")
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "fichier_source", "basename", "code_brut", "code_utilise", "sample_nb",
        "espece_trouvee", "classe_assignee", "statut", "destination"
    ]

    # 1) Fichier principal : uniquement les samples classés dans les 5 classes
    #    (c'est le livrable utile demandé).
    classes_rows = [r for r in report_rows if r["classe_assignee"]]
    classes_path = out_dir / f"samples_5_classes{suffix}.csv"
    with open(classes_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(classes_rows)

    # 2) Fichier secondaire : uniquement les codes vraiment inconnus (ni
    #    reconnus, ni génériques) -> les seuls à vérifier manuellement.
    unresolved_rows = [r for r in report_rows if r["statut"].startswith("NON_RESOLU") or r["statut"] == "FORMAT_NON_RECONNU"]
    unresolved_path = out_dir / f"a_verifier{suffix}.csv"
    with open(unresolved_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unresolved_rows)

    print()
    print("=== Résumé ===")
    print(f"Fichiers classés dans une des 5 classes : {n_classified}  -> {classes_path}")
    print(f"Fichiers exclus (autre espèce ou code générique ambigu) : {n_ignored_other_species}")
    print(f"Fichiers NON résolus (code inconnu, à vérifier manuellement) : {n_unresolved}  -> {unresolved_path}")
    if args.dry_run:
        print("\n(--dry-run actif : aucun fichier n'a été copié/déplacé)")


if __name__ == "__main__":
    main()