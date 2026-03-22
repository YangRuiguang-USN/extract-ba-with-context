# -*- coding: utf-8 -*-
"""
Extracteur de constructions en 把 - Version avec contexte v6
Architecture : deux passes, stratégie duale precision/recall

Passe 1 (jieba) — filtrage textuel rapide :
  • Présence de 把 dans la phrase
  • Absence de collocation figée (把握, 把手…)
  • Absence de mot de remplissage dans la zone NP2 (嗯/啊…)

Passe 2 (HanLP) — analyse syntaxique :
  • Identification via étiquette de dépendance 'ba' (plus fiable que POS jieba)
  • Extraction NP1/NP2/VP via SRL (ARG0/ARG1/PRED)

Stratégie duale (precision vs recall) :
  Couche 1 — hanlp_verified=TRUE :
    HanLP confirme 'ba' + SRL extrait NP2/VP
    → precision élevée, NP2/VP fiables
  Couche 2 — hanlp_verified=FALSE :
    HanLP ne trouve pas 'ba' (phrase trop complexe/orale) → jieba de secours
    Filtre NP2 ≤ 15 caractères appliqué pour éliminer les extractions jieba
    manifestement incorrectes (NP2 trop long = frontière mal détectée)
    → recall préservé, mais révision manuelle recommandée

Filtres de complétude (communs aux deux couches) :
  • NP2 non vide (absent de structure = pas une construction ba)
  • VP non vide (énoncé inachevé)

Colonnes CSV :
  source_file | timestamp | speaker_id | gender | dialect |
  context_before | sentence | context_after | NP1 | NP2 | VP | hanlp_verified

Compatibilité :
  Patch automatique HanLP 2.1.x + transformers 5.x
  (encode_plus supprimé dans transformers 5.x → redirigé vers _encode_plus)

Installation :
  pip install hanlp jieba
"""

import jieba.posseg as pseg
import csv
import os
import glob
import re


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

FIXED_COLLOCATIONS = [
    '把手', '把握', '把关', '把柄', '把酒', '把玩', '把守',
    '把持', '把脉', '把头', '把式', '把戏', '把子', '把臂'
]

FILLER_WORDS = {
    '嗯', '啊', '呃', '哦', '哎', '哟', '喂',
    '嗨', '哼', '诶', '唉', '哈'
}

FILLER_POS = {'e', 'y', 'o'}

# Seuil de longueur NP2 pour la couche jieba de secours
# NP2 > 15 caractères = frontière mal détectée par jieba → rejet
NP2_MAX_LEN_FALLBACK = 15


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT HANLP
# ══════════════════════════════════════════════════════════════════════════════

def load_hanlp():
    """
    Charge le modèle HanLP multitâche avec patch de compatibilité
    transformers 5.x. Retourne le pipeline ou None si indisponible.
    """
    try:
        import transformers
        if not hasattr(transformers.BertTokenizer, 'encode_plus'):
            transformers.BertTokenizer.encode_plus = (
                transformers.BertTokenizer._encode_plus
            )
        import hanlp
        print("Chargement du modèle HanLP...")
        pipeline = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
        )
        print("Modèle HanLP chargé.\n")
        return pipeline
    except ImportError:
        print("⚠  HanLP non installé → mode jieba uniquement.")
        print("   Pour activer HanLP : pip install hanlp\n")
        return None
    except Exception as e:
        print(f"⚠  Erreur HanLP : {e} → mode jieba uniquement.\n")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PASSE 1 — FILTRAGE JIEBA
# ══════════════════════════════════════════════════════════════════════════════

def has_filler_inside_ba(words, ba_position):
    """
    Vérifie si un mot de remplissage apparaît dans la zone NP2
    (entre 把 exclu et le premier verbe exclu).
    Frontière : position du premier verbe (plus fiable que la ponctuation
    en transcription orale).
    """
    search_end = min(ba_position + 13, len(words))
    verb_pos = search_end
    for k in range(ba_position + 1, search_end):
        if words[k].flag.startswith('v'):
            verb_pos = k
            break
    for k in range(ba_position + 1, verb_pos):
        w = words[k]
        if w.word in FILLER_WORDS and w.flag in FILLER_POS:
            return True
    return False


def jieba_extract_structure(words, ba_position):
    """
    Extraction NP1/NP2/VP par règles de position jieba.
    Utilisée comme secours si HanLP ne trouve pas de structure ba.
    """
    try:
        np1 = ''.join(w.word for w in words[:ba_position])
        verb_pos = next(
            (k for k in range(ba_position + 1, len(words))
             if words[k].flag.startswith('v')), -1
        )
        if verb_pos == -1:
            return np1, ''.join(w.word for w in words[ba_position + 1:]), ''
        np2 = ''.join(w.word for w in words[ba_position + 1:verb_pos])
        vp  = ''.join(w.word for w in words[verb_pos:])
        return np1, np2, vp
    except Exception:
        return '', '', ''


# ══════════════════════════════════════════════════════════════════════════════
# PASSE 2 — ANALYSE HANLP
# ══════════════════════════════════════════════════════════════════════════════

def clean_for_hanlp(sentence):
    """
    Nettoyage temporaire pour HanLP : supprime les marqueurs CTS
    ([+] [*] (SONANT) etc.) qui perturbent l'analyse syntaxique.
    Le texte original est toujours conservé dans le CSV.
    """
    cleaned = re.sub(r'\[.*?\]', '', sentence)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned.strip()


def hanlp_analyze(hanlp_pipeline, sentence):
    """Analyse HanLP sur la phrase nettoyée. Retourne None en cas d'erreur."""
    try:
        cleaned = clean_for_hanlp(sentence)
        if not cleaned:
            return None
        return hanlp_pipeline(cleaned)
    except Exception:
        return None


def hanlp_find_ba(hanlp_result):
    """
    Cherche 把 avec étiquette de dépendance 'ba'.
    Retourne l'index ou -1.
    L'étiquette 'ba' est spécifique au marqueur ba : plus fiable
    que la POS jieba 'p' qui couvre tous les prépositifs.
    """
    try:
        tokens = hanlp_result['tok/fine']
        deps   = hanlp_result['dep']
        for i, tok in enumerate(tokens):
            if tok == '把':
                _, dep_label = deps[i]
                if dep_label == 'ba':
                    return i
        return -1
    except Exception:
        return -1


def hanlp_extract_structure(hanlp_result, ba_idx):
    """
    Extrait NP1/NP2/VP via le SRL HanLP.

    Format observé :
      srl : [[('他', 'ARG0', 0, 1), ('那个苹果', 'ARG1', 2, 5), ('吃', 'PRED', 5, 6)]]
      ARG0 → agent (施事) = NP1
      ARG1 → patient (受事) = NP2
      PRED → prédicat = verbe noyau

    Les marqueurs aspectuels (了/过/着/完/好) immédiatement après PRED
    sont inclus dans le VP.

    Retourne (NP1, NP2, VP, succès:bool)
    """
    try:
        tokens = hanlp_result['tok/fine']
        srl    = hanlp_result['srl']

        for predicate_roles in srl:
            local_np1 = local_np2 = local_pred = ''
            local_pred_start = local_pred_end = -1

            for span_text, label, start, end in predicate_roles:
                if label == 'ARG0':
                    local_np1 = span_text
                elif label == 'ARG1':
                    local_np2 = span_text
                elif label == 'PRED':
                    local_pred       = span_text
                    local_pred_start = start
                    local_pred_end   = end

            # Vérifier que 把 précède le prédicat
            if local_np2 and local_pred_start > ba_idx:
                # VP = PRED + compléments + marqueurs aspectuels
                vp_parts = []
                for span_text, label, start, end in predicate_roles:
                    if label != 'ARG0' and start >= local_pred_start:
                        vp_parts.append((start, span_text))
                # Marqueurs aspectuels juste après PRED
                if local_pred_end >= 0:
                    for k in range(local_pred_end,
                                   min(local_pred_end + 3, len(tokens))):
                        if tokens[k] in {'了', '过', '着', '完', '好'}:
                            vp_parts.append((k, tokens[k]))
                        else:
                            break
                vp_parts.sort(key=lambda x: x[0])
                # Dédupliquer
                seen, dedup = set(), []
                for pos, text in vp_parts:
                    if pos not in seen:
                        seen.add(pos)
                        dedup.append(text)
                local_vp = ''.join(dedup)

                return local_np1, local_np2, local_vp if local_vp else local_pred, True

        return '', '', '', False

    except Exception:
        return '', '', '', False


# ══════════════════════════════════════════════════════════════════════════════
# PARSING CTS
# ══════════════════════════════════════════════════════════════════════════════

def parse_cts_line(line):
    """
    Parse une ligne CTS tab-séparée.
    Format : [timestamp] \\t speaker_id \\t genre,dialecte \\t texte
    Retourne dict ou None.
    """
    if not line.strip() or not line.startswith('['):
        return None
    parts = line.split('\t')
    if len(parts) < 4:
        return None
    gender_dialect = parts[2].strip().split(',')
    return {
        'timestamp':  parts[0].strip(),
        'speaker_id': parts[1].strip(),
        'gender':     gender_dialect[0].strip() if gender_dialect else '',
        'dialect':    gender_dialect[1].strip() if len(gender_dialect) > 1 else '',
        'sentence':   parts[3].strip()
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAITEMENT D'UN FICHIER
# ══════════════════════════════════════════════════════════════════════════════

def process_one_file(input_file, hanlp_pipeline, context_before=2, context_after=1):
    """
    Traite un fichier CTS en deux passes.

    Passe 1 (jieba) — tous les énoncés :
      • Présence de 把 + absence de collocation figée
      • Absence de filler dans la zone NP2

    Passe 2 (HanLP) — candidats jieba :
      • Identification via dep='ba'
      • Extraction NP1/NP2/VP via SRL (hanlp_verified=TRUE)
      • Sinon : jieba de secours avec filtre NP2 ≤ 15 (hanlp_verified=FALSE)

    Filtres de complétude (toujours appliqués) :
      • NP2 non vide
      • VP non vide

    Retourne (résultats, statistiques)
    """
    # Lecture
    lines = []
    for enc in ['utf-8', 'utf-8-sig', 'gbk']:
        try:
            with open(input_file, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except Exception:
            continue
    if not lines:
        return [], {}

    # Vérification format CTS
    if not any(l.strip().startswith('[') and '\t' in l for l in lines[:10]):
        print(f"  Format CTS non détecté, ignoré.")
        return [], {}

    parsed_lines = [p for l in lines for p in [parse_cts_line(l)] if p]
    results      = []
    source_name  = os.path.basename(input_file)

    stats = {
        'total_with_ba':         0,
        'rejected_colloc':       0,
        'rejected_filler':       0,
        'rejected_no_ba_hanlp':  0,
        'rejected_np2_too_long': 0,
        'rejected_empty_np2':    0,
        'rejected_empty_vp':     0,
        'hanlp_used':            0,
        'hanlp_fallback':        0,
        'extracted':             0,
    }

    for i, line_data in enumerate(parsed_lines):
        sentence = line_data['sentence']

        # ── Passe 1 : filtrage jieba ──────────────────────────────────────────

        if '把' not in sentence:
            continue
        stats['total_with_ba'] += 1

        if any(w in sentence for w in FIXED_COLLOCATIONS):
            stats['rejected_colloc'] += 1
            continue

        words = list(pseg.cut(sentence))

        jieba_ba_pos = -1
        for j, w in enumerate(words):
            if w.word == '把' and w.flag == 'p':
                if not has_filler_inside_ba(words, j):
                    jieba_ba_pos = j
                    break

        if jieba_ba_pos == -1:
            stats['rejected_filler'] += 1
            continue

        # ── Passe 2 : analyse HanLP ───────────────────────────────────────────

        np1 = np2 = vp = ''
        hanlp_verified = False

        if hanlp_pipeline is not None:
            hanlp_result = hanlp_analyze(hanlp_pipeline, sentence)

            if hanlp_result is not None:
                ba_idx = hanlp_find_ba(hanlp_result)

                if ba_idx != -1:
                    # Couche 1 : HanLP confirme 'ba' → SRL
                    np1_h, np2_h, vp_h, srl_ok = hanlp_extract_structure(
                        hanlp_result, ba_idx
                    )
                    if srl_ok and np2_h:
                        np1, np2, vp = np1_h, np2_h, vp_h
                        hanlp_verified = True
                        stats['hanlp_used'] += 1
                    else:
                        # SRL a échoué malgré le tag 'ba' → jieba de secours
                        np1, np2, vp = jieba_extract_structure(words, jieba_ba_pos)
                        stats['hanlp_fallback'] += 1
                else:
                    # Couche 2 : HanLP ne trouve pas 'ba' → jieba de secours
                    stats['rejected_no_ba_hanlp'] += 1
                    np1, np2, vp = jieba_extract_structure(words, jieba_ba_pos)
                    stats['hanlp_fallback'] += 1
            else:
                # HanLP a échoué sur cette phrase → jieba de secours
                np1, np2, vp = jieba_extract_structure(words, jieba_ba_pos)
                stats['hanlp_fallback'] += 1
        else:
            # Mode jieba uniquement
            np1, np2, vp = jieba_extract_structure(words, jieba_ba_pos)

        # ── Filtres de complétude ─────────────────────────────────────────────

        if not np2.strip():
            stats['rejected_empty_np2'] += 1
            continue

        if not vp.strip():
            stats['rejected_empty_vp'] += 1
            continue

        # Pour la couche jieba (hanlp_verified=FALSE) :
        # NP2 > 15 chars = frontière mal détectée → rejet
        if not hanlp_verified and len(np2.strip()) > NP2_MAX_LEN_FALLBACK:
            stats['rejected_np2_too_long'] += 1
            continue

        # ── Contexte ──────────────────────────────────────────────────────────

        ctx_before_list = [
            parsed_lines[k]['sentence']
            for k in range(max(0, i - context_before), i)
        ]
        ctx_after_list = [
            parsed_lines[k]['sentence']
            for k in range(i + 1, min(len(parsed_lines), i + context_after + 1))
        ]

        # ── Enregistrement ────────────────────────────────────────────────────

        results.append({
            'source_file':    source_name,
            'timestamp':      line_data['timestamp'],
            'speaker_id':     line_data['speaker_id'],
            'gender':         line_data['gender'],
            'dialect':        line_data['dialect'],
            'context_before': ' | '.join(ctx_before_list),
            'sentence':       sentence,
            'context_after':  ' | '.join(ctx_after_list),
            'NP1':            np1,
            'NP2':            np2,
            'VP':             vp,
            'hanlp_verified': hanlp_verified,
        })
        stats['extracted'] += 1

    return results, stats


# ══════════════════════════════════════════════════════════════════════════════
# TRAITEMENT EN BATCH
# ══════════════════════════════════════════════════════════════════════════════

def extract_ba_batch(input_folder, output_file, context_before=2, context_after=1):
    """Traite tous les fichiers CTS d'un dossier."""

    print("=" * 70)
    print("Extracteur de constructions en 把 - Version avec contexte v6")
    print("=" * 70)

    hanlp_pipeline = load_hanlp()

    pattern   = os.path.join(input_folder, '*.txt')
    txt_files = sorted(glob.glob(pattern))

    if not txt_files:
        print(f"Aucun fichier .txt trouvé dans : {input_folder}")
        return []

    mode = "jieba + HanLP" if hanlp_pipeline else "jieba uniquement"
    print(f"\nDossier  : {input_folder}")
    print(f"Fichiers : {len(txt_files)}")
    print(f"Mode     : {mode}")
    print(f"Contexte : {context_before} avant | {context_after} après")
    print(f"\nStratégie duale :")
    print(f"  Couche 1 [hanlp=TRUE]  : HanLP dep='ba' + SRL → precision élevée")
    print(f"  Couche 2 [hanlp=FALSE] : jieba de secours + NP2 ≤ {NP2_MAX_LEN_FALLBACK} chars → recall préservé")
    print(f"  Filtres communs : NP2 non vide, VP non vide\n")

    all_results = []
    total_stats = {k: 0 for k in [
        'total_with_ba', 'rejected_colloc', 'rejected_filler',
        'rejected_no_ba_hanlp', 'rejected_np2_too_long',
        'rejected_empty_np2', 'rejected_empty_vp',
        'hanlp_used', 'hanlp_fallback', 'extracted'
    ]}
    files_processed = files_with_results = 0

    for i, txt_file in enumerate(txt_files):
        filename = os.path.basename(txt_file)
        print(f"[{i+1}/{len(txt_files)}] {filename}")

        results, stats = process_one_file(
            txt_file, hanlp_pipeline, context_before, context_after
        )

        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

        if results:
            all_results.extend(results)
            files_with_results += 1
            print(
                f"  → {len(results)} extraite(s)"
                f" | HanLP: {stats.get('hanlp_used', 0)}"
                f" | jieba fallback: {stats.get('hanlp_fallback', 0)}"
                f" | NP2 trop long: {stats.get('rejected_np2_too_long', 0)}"
            )
        else:
            print(f"  → Aucune construction retenue")

        files_processed += 1

    # Sauvegarde CSV
    print("\n" + "=" * 70)
    print("Sauvegarde des résultats...")

    fieldnames = [
        'source_file', 'timestamp', 'speaker_id', 'gender', 'dialect',
        'context_before', 'sentence', 'context_after',
        'NP1', 'NP2', 'VP', 'hanlp_verified'
    ]

    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Rapport
    total = total_stats['total_with_ba']
    ext   = total_stats['extracted']
    hv    = total_stats['hanlp_used']
    fb    = total_stats['hanlp_fallback']

    print("=" * 70)
    print(f"Fichiers traités                   : {files_processed}")
    print(f"Fichiers avec résultats            : {files_with_results}")
    print()
    print(f"Phrases contenant 把               : {total}")
    print(f"  Rejetées (collocations)          : {total_stats['rejected_colloc']}")
    print(f"  Rejetées (filler dans NP2)       : {total_stats['rejected_filler']}")
    print(f"  Rejetées (NP2 vide)              : {total_stats['rejected_empty_np2']}")
    print(f"  Rejetées (VP vide)               : {total_stats['rejected_empty_vp']}")
    print(f"  Rejetées (NP2 trop long, jieba)  : {total_stats['rejected_np2_too_long']}")
    if total:
        print(f"  Retenues                         : {ext} ({ext/total*100:.1f}%)")
    print()
    if hanlp_pipeline:
        print(f"  Couche 1 hanlp_verified=TRUE     : {hv}"
              + (f" ({hv/ext*100:.1f}%)" if ext else ""))
        print(f"  Couche 2 hanlp_verified=FALSE    : {fb}"
              + (f" ({fb/ext*100:.1f}%)" if ext else ""))
    print()
    print(f"CSV sauvegardé : {output_file}")
    print(f"Colonnes : source_file | timestamp | speaker_id | gender | dialect |")
    print(f"           context_before | sentence | context_after | NP1 | NP2 | VP | hanlp_verified")
    print("=" * 70)

    if all_results:
        print("\n3 premiers exemples :\n")
        for k in range(min(3, len(all_results))):
            r = all_results[k]
            src = "HanLP ✓" if r['hanlp_verified'] else "jieba"
            print(f"  {k+1}. [{r['source_file']}] {r['speaker_id']} ({src})")
            if r['context_before']:
                print(f"     [AVANT]  {r['context_before'][:60]}")
            print(f"     [BA]     {r['sentence'][:70]}")
            if r['context_after']:
                print(f"     [APRÈS]  {r['context_after'][:60]}")
            print(f"     NP1=[{r['NP1']}] NP2=[{r['NP2']}] VP=[{r['VP']}]\n")
        if len(all_results) > 3:
            print(f"  ... et {len(all_results) - 3} de plus.")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Extracteur batch de constructions en 把 - v6")
    print("=" * 70)
    print("\nVersion finale — stratégie duale precision/recall :")
    print("  • Couche 1 [hanlp=TRUE]  : HanLP dep='ba' + SRL")
    print("  • Couche 2 [hanlp=FALSE] : jieba de secours, NP2 ≤ 15 chars")
    print("  • Suppression colonne is_subordinate")
    print("  • Filtres : NP2 vide, VP vide\n")

    input_folder = input(
        "Entrez le chemin du dossier contenant les fichiers CTS : "
    ).strip()

    if not os.path.isdir(input_folder):
        print(f"Dossier introuvable : {input_folder}")
    else:
        print("\nConfiguration du contexte :")
        try:
            ctx_before = int(input("Phrases AVANT (recommandé : 2) : ").strip() or "2")
            ctx_after  = int(input("Phrases APRÈS (recommandé : 1) : ").strip() or "1")
        except ValueError:
            ctx_before, ctx_after = 2, 1
            print("Valeurs par défaut utilisées (2 / 1).")

        folder_name = os.path.basename(input_folder.rstrip('/\\'))
        output_file = os.path.join(
            input_folder, f"{folder_name}_ba_with_context_v6.csv"
        )

        print(f"\nDossier d'entrée : {input_folder}")
        print(f"Fichier de sortie : {output_file}")
        print(f"Contexte : {ctx_before} avant | {ctx_after} après\n")
        print("Appuyez sur Entrée pour commencer...")
        input()

        extract_ba_batch(input_folder, output_file, ctx_before, ctx_after)

        print("\n" + "=" * 70)
        print("Traitement terminé !")
        print("=" * 70 + "\n")
