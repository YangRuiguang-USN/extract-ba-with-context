# -*- coding: utf-8 -*-
"""
Extracteur de constructions en 把 - Version avec contexte v7
Architecture : trois couches, intégration de l'analyse en dépendances HanLP

Changements par rapport à v6 :
  ─────────────────────────────────────────────────────────────────────────
  NOUVEAU — Couche 2 :校验 + 修正 par dépendances (dep)
  ─────────────────────────────────────────────────────────────────────────
  Quand la Couche 1 (SRL) échoue mais que HanLP confirme dep='ba',
  on tente d'extraire NP1/NP2/VP via l'arbre de dépendances plutôt que
  de tomber directement sur jieba.

  Logique dep pour une construction 把 :
    • 把 (dep='ba') → son head = verbe principal du VP
    • Les dépendants directs de 把 à droite = NP2
    • nsubj du verbe principal = NP1
    • Le verbe principal + ses compléments droits = VP

  La colonne hanlp_verified prend trois valeurs :
    TRUE_SRL  → Couche 1 : dep='ba' confirmé + SRL réussi
    TRUE_DEP  → Couche 2 : dep='ba' confirmé + extraction par dépendances
    FALSE     → Couche 3 : jieba de secours (comportement v6 inchangé)

  ─────────────────────────────────────────────────────────────────────────
  NOUVEAU — verbal_modifier extrait via dépendances
  ─────────────────────────────────────────────────────────────────────────
  Compléments de résultat, de degré, de direction et marqueurs aspectuels
  attachés au verbe principal, extraits depuis l'arbre dep et ajoutés
  dans une colonne séparée verbal_modifier.
  Labels dep couverts : rcomp, ccomp, dep, asp, loc, mmod, tmod

  ─────────────────────────────────────────────────────────────────────────
  NOUVEAU — fonction de校验 SRL par dep
  ─────────────────────────────────────────────────────────────────────────
  Même quand SRL réussit (Couche 1), les frontières NP2/VP sont
  vérifiées contre l'arbre dep. Si l'écart dépasse le seuil de
  tolérance, les valeurs dep remplacent les valeurs SRL.
  Contrôlé par ENABLE_DEP_CORRECTION (True par défaut).

Colonnes CSV :
  source_file | timestamp | speaker_id | gender | dialect |
  context_before | sentence | context_after |
  NP1 | NP2 | VP | verbal_modifier | hanlp_verified

Compatibilité : identique à v6 (patch transformers 5.x inclus)
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

# Seuil NP2 pour la couche jieba de secours
NP2_MAX_LEN_FALLBACK = 15

# Labels de dépendance couvrant les compléments verbaux (verbal_modifier)
VERBAL_MODIFIER_DEP_LABELS = {
    'rcomp',   # complément de résultat
    'ccomp',   # complément de clause
    'dep',     # dépendance non classée
    'asp',     # marqueur aspectuel
    'loc',     # complément locatif
    'mmod',    # modificateur modal
    'tmod',    # modificateur temporel
    'range',   # complément de portée (ex. 两遍)
    'eff',     # complément d'effet
    'pobj',    # objet de préposition dans VP
    'attr',    # attribut verbal
}

# Activer la校验/修正 des résultats SRL par l'arbre dep
ENABLE_DEP_CORRECTION = True

# Tolérance (en nombre de tokens) pour considérer SRL et dep concordants
DEP_CORRECTION_TOLERANCE = 1


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT HANLP
# ══════════════════════════════════════════════════════════════════════════════

def load_hanlp():
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
        return None
    except Exception as e:
        print(f"⚠  Erreur HanLP : {e} → mode jieba uniquement.\n")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PASSE 1 — FILTRAGE JIEBA
# ══════════════════════════════════════════════════════════════════════════════

def has_filler_inside_ba(words, ba_position):
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
    cleaned = re.sub(r'\[.*?\]', '', sentence)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned.strip()


def hanlp_analyze(hanlp_pipeline, sentence):
    try:
        cleaned = clean_for_hanlp(sentence)
        if not cleaned:
            return None
        return hanlp_pipeline(cleaned)
    except Exception:
        return None


def hanlp_find_ba(hanlp_result):
    """Retourne l'index de 把 avec dep='ba', ou -1."""
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


# ── Utilitaire dep ────────────────────────────────────────────────────────────

def _collect_subtree(token_idx, deps, tokens, visited=None):
    """
    Collecte récursivement tous les tokens dont la tête (directe ou
    indirecte) est token_idx. Retourne une liste triée d'indices.
    Protection contre les cycles (ne devrait pas arriver en dep bien formé).
    """
    if visited is None:
        visited = set()
    if token_idx in visited:
        return []
    visited.add(token_idx)

    result = [token_idx]
    for i, (head, _) in enumerate(deps):
        if head == token_idx and i not in visited:
            result.extend(_collect_subtree(i, deps, tokens, visited))
    return sorted(result)


def _tokens_to_str(indices, tokens):
    return ''.join(tokens[i] for i in indices)


# ── Extraction par dépendances ────────────────────────────────────────────────

def dep_extract_structure(hanlp_result, ba_idx):
    """
    Extrait NP1 / NP2 / VP / verbal_modifier via l'arbre de dépendances.

    Stratégie :
      1. ba_idx      → son head = verbe principal (vp_head)
      2. NP2         → sous-arbre du dépendant direct de 把 à droite de ba_idx
                       (label 'range' ou premier dépendant non-verbal)
      3. NP1         → token avec dep='nsubj' dont le head = vp_head
                       (ou nsubj de tout ancêtre de vp_head)
      4. VP          → vp_head + ses dépendants à droite (excl. NP2)
      5. verbal_mod  → dépendants du vp_head avec labels VERBAL_MODIFIER_DEP_LABELS

    Retourne (NP1, NP2, VP, verbal_modifier, succès:bool)
    """
    try:
        tokens = hanlp_result['tok/fine']
        deps   = hanlp_result['dep']   # liste de (head_index_1based, label)
                                        # HanLP utilise indices 1-based, 0 = root

        n = len(tokens)
        # Normaliser en 0-based (-1 = root)
        heads  = [h - 1 for h, _ in deps]
        labels = [l for _, l in deps]

        # 1. Trouver le head de 把 = verbe principal
        vp_head = heads[ba_idx]
        if vp_head < 0:
            return '', '', '', '', False  # 把 est root, cas anormal

        # 2. NP2 : dépendant direct de 把 (ou dépendant avec label 'range'/'ba')
        #    On prend le sous-arbre du premier dépendant direct de ba_idx
        #    situé à sa droite.
        np2_root = -1
        for i in range(ba_idx + 1, n):
            if heads[i] == ba_idx:
                np2_root = i
                break
        # Fallback : dépendant avec label 'range' dont head = vp_head
        if np2_root == -1:
            for i in range(ba_idx + 1, n):
                if heads[i] == vp_head and labels[i] in ('range', 'ba', 'dobj'):
                    np2_root = i
                    break

        if np2_root == -1:
            return '', '', '', '', False

        np2_indices = _collect_subtree(np2_root, list(zip(heads, labels)), tokens)
        np2 = _tokens_to_str(np2_indices, tokens)

        # 3. NP1 : chercher nsubj (ou nsubjpass) dont head = vp_head
        #    Si non trouvé, remonter jusqu'au root de la phrase.
        np1 = ''
        np1_root = -1
        candidate = vp_head
        for _ in range(5):  # au plus 5 niveaux de remontée
            for i in range(candidate):
                if heads[i] == candidate and labels[i] in ('nsubj', 'nsubjpass', 'top'):
                    np1_root = i
                    break
            if np1_root != -1:
                break
            if heads[candidate] < 0:
                break
            candidate = heads[candidate]

        if np1_root != -1:
            np1_indices = _collect_subtree(np1_root, list(zip(heads, labels)), tokens)
            # Exclure les tokens qui font partie de NP2 ou qui sont après 把
            np1_indices = [i for i in np1_indices if i < ba_idx]
            np1 = _tokens_to_str(np1_indices, tokens)

        # 4. VP : vp_head + dépendants à droite sauf NP2
        np2_set = set(np2_indices)
        vp_indices = _collect_subtree(vp_head, list(zip(heads, labels)), tokens)
        vp_indices = [i for i in vp_indices if i not in np2_set and i > ba_idx]
        # S'assurer que vp_head lui-même est inclus
        if vp_head not in vp_indices:
            vp_indices = sorted([vp_head] + vp_indices)
        vp = _tokens_to_str(sorted(vp_indices), tokens)

        # 5. verbal_modifier : dépendants de vp_head avec labels verbaux
        vm_indices = []
        for i, (h, lbl) in enumerate(zip(heads, labels)):
            if h == vp_head and lbl in VERBAL_MODIFIER_DEP_LABELS and i > vp_head:
                vm_indices.extend(_collect_subtree(i, list(zip(heads, labels)), tokens))
        vm_indices = sorted(set(vm_indices) - {vp_head})
        verbal_modifier = _tokens_to_str(vm_indices, tokens) if vm_indices else ''

        if not np2 or not vp:
            return np1, np2, vp, verbal_modifier, False

        return np1, np2, vp, verbal_modifier, True

    except Exception:
        return '', '', '', '', False


# ── Extraction par SRL (v6) ───────────────────────────────────────────────────

def hanlp_extract_structure_srl(hanlp_result, ba_idx):
    """
    Extrait NP1/NP2/VP via le SRL HanLP (logique v6 inchangée).
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

            if local_np2 and local_pred_start > ba_idx:
                vp_parts = []
                for span_text, label, start, end in predicate_roles:
                    if label != 'ARG0' and start >= local_pred_start:
                        vp_parts.append((start, span_text))
                if local_pred_end >= 0:
                    for k in range(local_pred_end,
                                   min(local_pred_end + 3, len(tokens))):
                        if tokens[k] in {'了', '过', '着', '完', '好'}:
                            vp_parts.append((k, tokens[k]))
                        else:
                            break
                vp_parts.sort(key=lambda x: x[0])
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


# ──校验 SRL par dep ──────────────────────────────────────────────────────────

def _char_boundary_from_tokens(start_tok, end_tok, tokens):
    """Convertit des indices token en offset caractère dans la phrase concaténée."""
    chars_before = sum(len(tokens[i]) for i in range(start_tok))
    chars_span   = sum(len(tokens[i]) for i in range(start_tok, end_tok))
    return chars_before, chars_before + chars_span


def dep_correct_srl(np2_srl, vp_srl, np2_dep, vp_dep):
    """
    Compare les résultats SRL et dep pour NP2 et VP.
    Si les longueurs divergent au-delà du seuil, on préfère dep.
    Retourne (np2, vp, corrigé:bool)
    """
    if not ENABLE_DEP_CORRECTION:
        return np2_srl, vp_srl, False

    np2_diff = abs(len(np2_srl) - len(np2_dep))
    vp_diff  = abs(len(vp_srl)  - len(vp_dep))

    if (np2_diff > DEP_CORRECTION_TOLERANCE or
            vp_diff > DEP_CORRECTION_TOLERANCE):
        # Préférer dep si disponible
        np2_final = np2_dep if np2_dep else np2_srl
        vp_final  = vp_dep  if vp_dep  else vp_srl
        return np2_final, vp_final, True

    return np2_srl, vp_srl, False


# ══════════════════════════════════════════════════════════════════════════════
# PARSING CTS
# ══════════════════════════════════════════════════════════════════════════════

def parse_cts_line(line):
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
    Traite un fichier CTS en deux passes, trois couches.

    Couche 1 — hanlp_verified='TRUE_SRL' :
      HanLP dep='ba' confirmé + SRL réussi
      (+校验/修正 par dep si ENABLE_DEP_CORRECTION=True)

    Couche 2 — hanlp_verified='TRUE_DEP' :
      HanLP dep='ba' confirmé + SRL échoué → extraction par dépendances

    Couche 3 — hanlp_verified='FALSE' :
      jieba fallback (comportement v6 inchangé)
    """
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
        'layer1_srl':            0,   # Couche 1 : SRL réussi
        'layer1_srl_corrected':  0,   # Couche 1 : SRL corrigé par dep
        'layer2_dep':            0,   # Couche 2 : dep fallback
        'layer3_jieba':          0,   # Couche 3 : jieba fallback
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
        verbal_modifier = ''
        hanlp_verified  = 'FALSE'

        if hanlp_pipeline is not None:
            hanlp_result = hanlp_analyze(hanlp_pipeline, sentence)

            if hanlp_result is not None:
                ba_idx = hanlp_find_ba(hanlp_result)

                if ba_idx != -1:
                    # ── Couche 1 : SRL ───────────────────────────────────────
                    np1_s, np2_s, vp_s, srl_ok = hanlp_extract_structure_srl(
                        hanlp_result, ba_idx
                    )

                    # Toujours extraire dep pour校验 et verbal_modifier
                    np1_d, np2_d, vp_d, vm_d, dep_ok = dep_extract_structure(
                        hanlp_result, ba_idx
                    )

                    if srl_ok and np2_s:
                        #校验 SRL par dep
                        np2_final, vp_final, was_corrected = dep_correct_srl(
                            np2_s, vp_s, np2_d, vp_d
                        )
                        np1 = np1_s if np1_s else np1_d
                        np2 = np2_final
                        vp  = vp_final
                        verbal_modifier = vm_d
                        hanlp_verified  = 'TRUE_SRL'
                        if was_corrected:
                            stats['layer1_srl_corrected'] += 1
                        else:
                            stats['layer1_srl'] += 1

                    elif dep_ok and np2_d:
                        # ── Couche 2 : dep fallback ───────────────────────────
                        np1 = np1_d
                        np2 = np2_d
                        vp  = vp_d
                        verbal_modifier = vm_d
                        hanlp_verified  = 'TRUE_DEP'
                        stats['layer2_dep'] += 1

                    else:
                        # SRL et dep ont tous les deux échoué → jieba
                        np1, np2, vp = jieba_extract_structure(words, jieba_ba_pos)
                        hanlp_verified = 'FALSE'
                        stats['layer3_jieba'] += 1

                else:
                    # HanLP ne trouve pas dep='ba' → jieba
                    stats['rejected_no_ba_hanlp'] += 1
                    np1, np2, vp = jieba_extract_structure(words, jieba_ba_pos)
                    hanlp_verified = 'FALSE'
                    stats['layer3_jieba'] += 1

            else:
                # HanLP a échoué → jieba
                np1, np2, vp = jieba_extract_structure(words, jieba_ba_pos)
                hanlp_verified = 'FALSE'
                stats['layer3_jieba'] += 1

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

        if hanlp_verified == 'FALSE' and len(np2.strip()) > NP2_MAX_LEN_FALLBACK:
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
            'source_file':      source_name,
            'timestamp':        line_data['timestamp'],
            'speaker_id':       line_data['speaker_id'],
            'gender':           line_data['gender'],
            'dialect':          line_data['dialect'],
            'context_before':   ' | '.join(ctx_before_list),
            'sentence':         sentence,
            'context_after':    ' | '.join(ctx_after_list),
            'NP1':              np1,
            'NP2':              np2,
            'VP':               vp,
            'verbal_modifier':  verbal_modifier,
            'hanlp_verified':   hanlp_verified,
        })
        stats['extracted'] += 1

    return results, stats


# ══════════════════════════════════════════════════════════════════════════════
# TRAITEMENT EN BATCH
# ══════════════════════════════════════════════════════════════════════════════

def extract_ba_batch(input_folder, output_file, context_before=2, context_after=1):
    print("=" * 70)
    print("Extracteur de constructions en 把 - Version avec contexte v7")
    print("=" * 70)

    hanlp_pipeline = load_hanlp()

    pattern   = os.path.join(input_folder, '*.txt')
    txt_files = sorted(glob.glob(pattern))

    if not txt_files:
        print(f"Aucun fichier .txt trouvé dans : {input_folder}")
        return []

    mode = "jieba + HanLP (SRL + dep)" if hanlp_pipeline else "jieba uniquement"
    print(f"\nDossier  : {input_folder}")
    print(f"Fichiers : {len(txt_files)}")
    print(f"Mode     : {mode}")
    print(f"Contexte : {context_before} avant | {context_after} après")
    print(f"\nStratégie trois couches :")
    print(f"  Couche 1 [TRUE_SRL]  : dep='ba' + SRL (校验 par dep : {ENABLE_DEP_CORRECTION})")
    print(f"  Couche 2 [TRUE_DEP]  : dep='ba' + extraction dépendances")
    print(f"  Couche 3 [FALSE]     : jieba fallback, NP2 ≤ {NP2_MAX_LEN_FALLBACK} chars")
    print(f"  Filtres communs : NP2 non vide, VP non vide\n")

    all_results = []
    total_stats = {k: 0 for k in [
        'total_with_ba', 'rejected_colloc', 'rejected_filler',
        'rejected_no_ba_hanlp', 'rejected_np2_too_long',
        'rejected_empty_np2', 'rejected_empty_vp',
        'layer1_srl', 'layer1_srl_corrected', 'layer2_dep', 'layer3_jieba',
        'extracted'
    ]}
    files_processed = files_with_results = 0

    for idx, txt_file in enumerate(txt_files):
        filename = os.path.basename(txt_file)
        print(f"[{idx+1}/{len(txt_files)}] {filename}")

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
                f" | SRL: {stats.get('layer1_srl', 0)}"
                f" | SRL+corr: {stats.get('layer1_srl_corrected', 0)}"
                f" | dep: {stats.get('layer2_dep', 0)}"
                f" | jieba: {stats.get('layer3_jieba', 0)}"
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
        'NP1', 'NP2', 'VP', 'verbal_modifier', 'hanlp_verified'
    ]

    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Rapport
    total = total_stats['total_with_ba']
    ext   = total_stats['extracted']
    l1    = total_stats['layer1_srl'] + total_stats['layer1_srl_corrected']
    l1c   = total_stats['layer1_srl_corrected']
    l2    = total_stats['layer2_dep']
    l3    = total_stats['layer3_jieba']

    print("=" * 70)
    print(f"Fichiers traités                        : {files_processed}")
    print(f"Fichiers avec résultats                 : {files_with_results}")
    print()
    print(f"Phrases contenant 把                    : {total}")
    print(f"  Rejetées (collocations)               : {total_stats['rejected_colloc']}")
    print(f"  Rejetées (filler dans NP2)            : {total_stats['rejected_filler']}")
    print(f"  Rejetées (NP2 vide)                   : {total_stats['rejected_empty_np2']}")
    print(f"  Rejetées (VP vide)                    : {total_stats['rejected_empty_vp']}")
    print(f"  Rejetées (NP2 trop long, jieba)       : {total_stats['rejected_np2_too_long']}")
    if total:
        print(f"  Retenues                              : {ext} ({ext/total*100:.1f}%)")
    print()
    if hanlp_pipeline:
        print(f"  Couche 1 TRUE_SRL (SRL pur)          : {total_stats['layer1_srl']}"
              + (f" ({total_stats['layer1_srl']/ext*100:.1f}%)" if ext else ""))
        print(f"  Couche 1 TRUE_SRL (corrigé par dep)  : {l1c}"
              + (f" ({l1c/ext*100:.1f}%)" if ext else ""))
        print(f"  Couche 2 TRUE_DEP (dep fallback)     : {l2}"
              + (f" ({l2/ext*100:.1f}%)" if ext else ""))
        print(f"  Couche 3 FALSE    (jieba fallback)   : {l3}"
              + (f" ({l3/ext*100:.1f}%)" if ext else ""))
    print()
    print(f"CSV sauvegardé : {output_file}")
    print(f"Colonnes : source_file | timestamp | speaker_id | gender | dialect |")
    print(f"           context_before | sentence | context_after |")
    print(f"           NP1 | NP2 | VP | verbal_modifier | hanlp_verified")
    print("=" * 70)

    if all_results:
        print("\n3 premiers exemples :\n")
        for k in range(min(3, len(all_results))):
            r = all_results[k]
            print(f"  {k+1}. [{r['source_file']}] {r['speaker_id']} [{r['hanlp_verified']}]")
            if r['context_before']:
                print(f"     [AVANT]  {r['context_before'][:60]}")
            print(f"     [BA]     {r['sentence'][:70]}")
            if r['context_after']:
                print(f"     [APRÈS]  {r['context_after'][:60]}")
            print(f"     NP1=[{r['NP1']}] NP2=[{r['NP2']}] VP=[{r['VP']}]")
            if r['verbal_modifier']:
                print(f"     verbal_modifier=[{r['verbal_modifier']}]")
            print()
        if len(all_results) > 3:
            print(f"  ... et {len(all_results) - 3} de plus.")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Extracteur batch de constructions en 把 - v7")
    print("=" * 70)
    print("\nVersion v7 — trois couches SRL / dep / jieba :")
    print("  • Couche 1 [TRUE_SRL]  : HanLP dep='ba' + SRL (+ 校验 par dep)")
    print("  • Couche 2 [TRUE_DEP]  : HanLP dep='ba' + extraction dépendances")
    print("  • Couche 3 [FALSE]     : jieba fallback, NP2 ≤ 15 chars")
    print("  • Nouvelle colonne verbal_modifier")
    print("  • hanlp_verified : TRUE_SRL / TRUE_DEP / FALSE\n")

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
            input_folder, f"{folder_name}_ba_with_context_v7.csv"
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
