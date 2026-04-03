# -*- coding: utf-8 -*-
"""
merge_manual_v7.py
──────────────────
Fusionne les annotations manuelles (v6 annotée) avec les nouveaux
résultats v7, en respectant les priorités suivantes :

  PRIORITÉ 1 — Lignes manuellement annotées (qualite = 0/1/?)
    → les valeurs manuelles (NP1/NP2/VP/verbal_modifier/sentence) sont
      conservées comme référence dans les colonnes principales
    → si v7 a un résultat différent, les valeurs v7 sont stockées dans
      des colonnes v7_* séparées, et diff_fields indique les champs divergents
    → la décision finale reste à l'utilisateur (colonne keep_manual : 1 par défaut)

  PRIORITÉ 2 — Lignes NON annotées présentes dans v6
    → remplacées par le résultat v7 correspondant (même clé)
    → si v7 n'a pas de correspondance, on garde la ligne v6

  PRIORITÉ 3 — Nouvelles lignes de v7 absentes de v6
    → ajoutées à la fin

Clé d'identification : (source_file, timestamp, sentence_original)
  • sentence_original = colonne 'sentence_original' dans v6 annoté
  • dans v7, cette colonne s'appelle 'sentence' (texte brut avant nettoyage)

Colonnes de sortie :
  Colonnes principales (valeurs manuelles pour les annotées, v7 sinon) +
  colonnes de diff pour les annotées :
    v7_NP1 / v7_NP2 / v7_VP / v7_verbal_modifier  → valeurs v7 brutes
    diff_fields   → liste des champs divergents, ex: "NP2,VP"
    keep_manual   → 1 par défaut (garder manuel), mettre 0 pour adopter v7
    merged_from   → traçabilité : manual_nodiff / manual_diff / v7_updated /
                                  v6_kept / v7_new

Usage :
  python merge_manual_v7.py \\
      --manual   chemin/vers/v6_annote.csv \\
      --v7       chemin/vers/v7_results.csv \\
      --output   chemin/vers/merged.csv
"""

import csv
import argparse
import os
import sys


# ══════════════════════════════════════════════════════════════════════════════
# COLONNES
# ══════════════════════════════════════════════════════════════════════════════

# Colonnes héritées du fichier manuel (v6 annoté)
MANUAL_COLS = [
    'source_file', 'timestamp', 'speaker_id', 'gender', 'dialect',
    'context_before', 'sentence_original', 'context_after',
    'qualite ( 1 à conserver, 0 à supprimer, ? à vérifier)',
    'reason',
    'sentence',     # version nettoyée / corrigée manuellement
    'NP1', 'NP2', 'VP',
    'verbal_modifier',
    'verbal_modifier_position',
    'verbal_modifier_type',
    'hanlp_verified',
]

# Colonnes finales dans le CSV de sortie (ordre voulu)
OUTPUT_COLS = [
    'source_file', 'timestamp', 'speaker_id', 'gender', 'dialect',
    'context_before', 'sentence_original', 'context_after',
    'qualite ( 1 à conserver, 0 à supprimer, ? à vérifier)',
    'reason',
    'sentence',
    'NP1', 'NP2', 'VP',
    'verbal_modifier',
    'verbal_modifier_position',
    'verbal_modifier_type',
    'hanlp_verified',
    # Colonnes de diff (remplies uniquement pour les lignes annotées)
    'v7_NP1',
    'v7_NP2',
    'v7_VP',
    'v7_verbal_modifier',
    'diff_fields',        # ex: "NP2,VP" — vide si aucune divergence
    'keep_manual',        # 1 = garder manuel (defaut), 0 = adopter v7
    # Tracabilite
    'merged_from',        # manual_nodiff / manual_diff / v7_updated / v6_kept / v7_new
]

# Champs a comparer entre version manuelle et v7
DIFF_FIELDS = ['NP1', 'NP2', 'VP', 'verbal_modifier']

ANNOTATED_VALUES = {'0', '1', '?'}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def make_key(source_file, timestamp, sentence_original):
    """Clé de déduplication : (fichier source, horodatage, phrase originale)."""
    return (
        source_file.strip(),
        timestamp.strip(),
        sentence_original.strip(),
    )


def read_csv(path, encoding='utf-8-sig'):
    """Lit un CSV et retourne (fieldnames, rows)."""
    for enc in [encoding, 'utf-8', 'gbk']:
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return reader.fieldnames or [], rows
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Impossible de lire : {path}")


def write_csv(path, rows, fieldnames):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_manual_row(row):
    """
    Normalise une ligne du fichier manuel vers le schéma OUTPUT_COLS.
    Les colonnes vides supplémentaires (colonnes sans nom) sont ignorées.
    """
    out = {col: '' for col in OUTPUT_COLS}
    for col in MANUAL_COLS:
        if col in row:
            out[col] = row[col]
    # sentence_original : dans v6 annoté c'est déjà la bonne colonne
    # Si absent, utiliser 'sentence' comme fallback
    if not out['sentence_original'] and 'sentence' in row:
        out['sentence_original'] = row['sentence']
    return out


def normalize_v7_row(row):
    """
    Normalise une ligne v7 vers le schéma OUTPUT_COLS.
    Dans v7 : 'sentence' = texte brut (= sentence_original dans v6)
    """
    out = {col: '' for col in OUTPUT_COLS}
    # Mapping v7 → schéma de sortie
    mapping = {
        'source_file':    'source_file',
        'timestamp':      'timestamp',
        'speaker_id':     'speaker_id',
        'gender':         'gender',
        'dialect':        'dialect',
        'context_before': 'context_before',
        'context_after':  'context_after',
        'NP1':            'NP1',
        'NP2':            'NP2',
        'VP':             'VP',
        'verbal_modifier':'verbal_modifier',
        'hanlp_verified': 'hanlp_verified',
    }
    for v7_col, out_col in mapping.items():
        if v7_col in row:
            out[out_col] = row[v7_col]

    # Dans v7, 'sentence' = texte brut = sentence_original
    v7_sentence = row.get('sentence', '')
    out['sentence_original'] = v7_sentence
    out['sentence']          = v7_sentence   # pas de version manuelle

    return out


# ══════════════════════════════════════════════════════════════════════════════
# FUSION
# ══════════════════════════════════════════════════════════════════════════════

def merge(manual_path, v7_path, output_path):

    print("=" * 65)
    print("Fusion annotations manuelles + résultats v7")
    print("=" * 65)

    # ── Lecture ───────────────────────────────────────────────────────────────
    _, manual_rows = read_csv(manual_path)
    _, v7_rows     = read_csv(v7_path)

    print(f"\nFichier manuel  : {os.path.basename(manual_path)}")
    print(f"  Lignes totales : {len(manual_rows)}")

    # Séparer lignes annotées / non annotées dans le fichier manuel
    annotated     = []
    not_annotated = []
    for row in manual_rows:
        q = row.get('qualite ( 1 à conserver, 0 à supprimer, ? à vérifier)', '').strip()
        if q in ANNOTATED_VALUES:
            annotated.append(row)
        else:
            not_annotated.append(row)

    print(f"  Annotées (0/1/?) : {len(annotated)}")
    print(f"  Non annotées     : {len(not_annotated)}")
    print(f"\nFichier v7      : {os.path.basename(v7_path)}")
    print(f"  Lignes totales : {len(v7_rows)}")

    # ── Index v7 par clé ─────────────────────────────────────────────────────
    v7_index = {}
    for row in v7_rows:
        key = make_key(
            row.get('source_file', ''),
            row.get('timestamp', ''),
            row.get('sentence', '')      # dans v7, 'sentence' = phrase originale
        )
        v7_index[key] = row

    # ── Index des clés déjà couvertes par les annotations manuelles ──────────
    annotated_keys = set()
    for row in annotated:
        key = make_key(
            row.get('source_file', ''),
            row.get('timestamp', ''),
            row.get('sentence_original', row.get('sentence', ''))
        )
        annotated_keys.add(key)

    # ── Construction du résultat ─────────────────────────────────────────────
    result = []
    diff_count = 0

    # 1. Lignes annotées manuellement → conserver + détecter divergences v7
    for row in annotated:
        out = normalize_manual_row(row)

        # Chercher la ligne v7 correspondante
        key = make_key(
            row.get('source_file', ''),
            row.get('timestamp', ''),
            row.get('sentence_original', row.get('sentence', ''))
        )
        v7_row = v7_index.get(key)

        if v7_row is not None:
            # Comparer champ par champ
            divergent = []
            for field in DIFF_FIELDS:
                manual_val = (row.get(field) or '').strip()
                v7_val     = (v7_row.get(field) or '').strip()
                if manual_val != v7_val:
                    divergent.append(field)
                    out[f'v7_{field}'] = v7_val   # stocker valeur v7

            if divergent:
                out['diff_fields']  = ','.join(divergent)
                out['keep_manual']  = '1'   # par défaut : garder le manuel
                out['merged_from']  = 'manual_diff'
                diff_count += 1
            else:
                out['merged_from']  = 'manual_nodiff'
        else:
            # Pas de correspondance v7 → aucune comparaison possible
            out['merged_from'] = 'manual_nodiff'

        result.append(out)

    # 2. Lignes non annotées → remplacer par v7 si disponible
    updated = skipped = kept_v6 = 0
    for row in not_annotated:
        key = make_key(
            row.get('source_file', ''),
            row.get('timestamp', ''),
            row.get('sentence_original', row.get('sentence', ''))
        )
        if key in v7_index:
            out = normalize_v7_row(v7_index[key])
            out['merged_from'] = 'v7_updated'
            result.append(out)
            updated += 1
        else:
            # v7 n'a pas cette phrase → garder v6
            out = normalize_manual_row(row)
            out['merged_from'] = 'v7_updated'   # sera vide, on marque v6_kept
            out['merged_from'] = 'v6_kept'
            result.append(out)
            kept_v6 += 1

    # 3. Nouvelles lignes v7 absentes de v6
    v6_all_keys = set()
    for row in manual_rows:
        key = make_key(
            row.get('source_file', ''),
            row.get('timestamp', ''),
            row.get('sentence_original', row.get('sentence', ''))
        )
        v6_all_keys.add(key)

    new_rows = 0
    for row in v7_rows:
        key = make_key(
            row.get('source_file', ''),
            row.get('timestamp', ''),
            row.get('sentence', '')
        )
        if key not in v6_all_keys:
            out = normalize_v7_row(row)
            out['merged_from'] = 'v7_new'
            result.append(out)
            new_rows += 1

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    write_csv(output_path, result, OUTPUT_COLS)

    # ── Rapport ───────────────────────────────────────────────────────────────
    print(f"\nRésultat de la fusion :")
    print(f"  Annotées — sans divergence v7        : {len(annotated) - diff_count}")
    print(f"  Annotées — avec divergence v7        : {diff_count}  ← à vérifier")
    print(f"  Mises à jour par v7                  : {updated}")
    print(f"  Gardées depuis v6 (absentes de v7)   : {kept_v6}")
    print(f"  Nouvelles lignes v7                  : {new_rows}")
    print(f"  ──────────────────────────────────────────")
    print(f"  TOTAL                                : {len(result)}")
    print(f"\nCSV sauvegardé : {output_path}")
    print(f"\nColonne merged_from :")
    print(f"  manual_nodiff  → manuel = v7, aucun problème")
    print(f"  manual_diff    → divergence détectée, vérifier diff_fields + v7_*")
    print(f"  v7_updated     → ligne non annotée, mise à jour par v7")
    print(f"  v6_kept        → ligne non annotée, absente de v7")
    print(f"  v7_new         → nouvelle phrase détectée par v7 uniquement")
    print(f"\nPour les lignes manual_diff :")
    print(f"  keep_manual=1  → valeurs manuelles conservées dans NP1/NP2/VP/VM")
    print(f"  keep_manual=0  → mettre 0 pour signaler que v7 a raison (à traiter manuellement)")
    print("=" * 65)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fusionne annotations manuelles v6 + résultats v7"
    )
    parser.add_argument('--manual', required=True,
                        help='CSV v6 avec annotations manuelles')
    parser.add_argument('--v7',     required=True,
                        help='CSV produit par extract_ba_with_context_v7.py')
    parser.add_argument('--output', required=True,
                        help='Chemin du CSV de sortie fusionné')
    args = parser.parse_args()

    for path in [args.manual, args.v7]:
        if not os.path.isfile(path):
            print(f"Fichier introuvable : {path}")
            sys.exit(1)

    merge(args.manual, args.v7, args.output)


if __name__ == '__main__':
    main()
