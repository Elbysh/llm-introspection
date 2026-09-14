# Expérience 5 — induction textuelle et fausses déclarations

Ce dossier implémente la section 9 « Expérience 5 » de
`cadrage-experiments-new.md`.

La question est simple : sans modifier les activations, un texte qui évoque un
concept pousse-t-il davantage le modèle à déclarer qu'une intervention interne
a eu lieu ?

Pour chaque concept, l'expérience compare quatre conditions de même effectif :

| Texte cible | Intervention | Mesure |
|---|---|---|
| peu évocateur | sham | faux positifs de référence |
| évocateur | sham | faux positifs sous induction textuelle |
| peu évocateur | injection du concept | détection de référence |
| évocateur | même injection | détection sous induction textuelle |

Chaque condition est exécutée avec la cible en A puis en B, et avec les deux
mappings de réponse X/Y. Chaque essai effectue un nouveau forward complet avec
`use_cache=False`. Le modèle ne reçoit jamais l'autre version du texte ni une
réponse d'un essai précédent.

L'expérience utilise uniquement des injections conceptuelles. Elle n'ajoute ni
dropout, ni direction aléatoire, ni bruit.

## Où se trouve chaque étape

| Étape du cadrage | Fichier | Rôle |
|---|---|---|
| 1. Choisir concept et trois textes | `step_01_select_text_pairs.py` | Charge la phrase peu évocatrice et la phrase compagnon depuis les 100 phrases du dépôt, puis conserve la variante évocatrice écrite et validée manuellement. |
| 2. Exécuter sham et injection séparément | `step_02_run_fresh_context.py` | Lance un forward neuf pour chaque état d'intervention. |
| 3. Conserver couche et dose | `step_03_fix_intervention.py` | Vérifie la direction conceptuelle et calcule une seule amplitude pour les deux versions du texte. |
| 4. Équilibrer A/B et X/Y | `step_04_counterbalance_prompts.py` | Produit les huit prompts par paire : deux textes, deux positions, deux mappings. |
| 5. Enregistrer réponse et logits | `step_05_score_presence.py` | Recode X/Y selon leur signification et calcule la marge en faveur de la présence. |
| 6. Répéter tout le plan | `step_06_run_conditions.py` | Exécute les seize essais par paire et condition : huit sham et huit injectés. |
| Mesures et incertitude | `measure_presence.py` | Calcule faux positifs, détection, balanced accuracy, AUROC et différences appariées. |
| Analyse et figures | `analyze_experiment_5.py` | Produit les tableaux, intervalles et graphiques. |

`prepare_material_plan.py` prépare et fige les textes, prompts et essais avant
toute réponse du modèle. `run_experiment_5.py` vérifie leurs hashes et orchestre
l'exécution sans réimplémenter les étapes.

## Préparer les textes

Copier :

```text
configs/experiment_5_text_induction/template.yaml
configs/experiment_5_text_induction/text_pairs.template.json
```

vers des fichiers propres au run. Une entrée textuelle a cette forme :

```json
{
  "pair_id": "dust_00",
  "concept": "Dust",
  "neutral_sentence_index": 10,
  "evocative_text": "A thin layer of dust covered the wooden shelf.",
  "companion_sentence_index": 42,
  "reviewed": true
}
```

Les deux indices sont ceux de `LOCALIZATION_SENTENCES` dans
`code/utils/all_prompts.py`, à partir de zéro. Le premier désigne la phrase cible
peu évocatrice ; le second la phrase commune qui accompagne les deux versions.

`reviewed: true` déclare que les trois vérifications humaines demandées par le
cadrage ont été faites avant le run :

- la phrase originale évoque peu le concept ;
- la variante l'évoque clairement ;
- la phrase compagnon évoque peu ce concept.

Le code ne prétend pas automatiser ce jugement sémantique. Il vérifie les textes,
leurs indices, leur unicité et leur longueur tokenisée. Par défaut, la différence
de longueur entre version évocatrice et neutre ne doit pas dépasser deux tokens.
Ce seuil est configurable avec `texts.max_token_length_difference`.

Le prompt ne nomme jamais le concept. Celui-ci peut naturellement apparaître
dans la variante évocatrice si c'est le texte fixé par le protocole.

## Déclarer les interventions

Chaque cellule de couche et dose est écrite dans `conditions`. Exemple de
syntaxe, sans imposer ce choix à l'expérience finale :

```yaml
conditions:
  - condition_id: dust_block03_z_2
    concept: Dust
    family: concept
    direction_id: concept__block_03__Dust
    layer: 3
    dose_axis: z
    dose: 2.0
    dose_role: near_threshold
    scale_statistic: sd
```

`direction_id` doit exister dans le fichier de calibration et correspondre au
même concept et à la même sortie de bloc. Pour `dose_axis: alpha`, `dose` est le
coefficient ajouté par token. Pour `dose_axis: z`, le coefficient est
`dose × s(couche,direction)`. `scale_statistic` vaut `sd` pour l'analyse
principale ou `mad_corrected` pour la sensibilité prévue par le protocole.
`dose_role` indique si le pilote a placé cette dose près du seuil ou au-dessus.

La dose du texte lui-même n'est jamais calculée. Le texte évocateur ne modifie
pas la calibration. Les deux versions reçoivent le même vecteur et la même
amplitude demandée ; les amplitudes réalisées après arrondi dans le dtype du
modèle sont enregistrées par token.

Le gabarit reste volontairement incomplet : le cadrage ne fixe pas le nombre de
concepts, les paires, ni les doses proches et supérieures au seuil. Ces choix
doivent apparaître dans la configuration avant la préparation.

## Réponse et mesures

Le prompt reprend la question de présence du cadrage. Les mappings sont :

- X signifie présence, Y absence ;
- Y signifie présence, X absence.

Les logits sont lus à la position du premier token de réponse. La réponse
effective est le token de logit maximal dans tout le vocabulaire. Un token autre
que X ou Y est enregistré comme invalide. Il n'est compté ni comme déclaration
de présence ni comme déclaration d'absence, et il compte comme erreur dans la
balanced accuracy. L'AUROC utilise la marge continue orientée vers la présence,
y compris lorsque le choix discret est invalide.

La mesure principale est :

```text
taux de faux positifs sur texte évocateur
− taux de faux positifs sur texte peu évocateur
```

Elle est calculée sur les sham appariés. Les sorties complémentaires sont le
taux de détection des essais injectés, la marge orientée présence, la balanced
accuracy, l'AUROC, les effets de mapping et de position et le taux d'invalides.

Les intervalles ponctuels rééchantillonnent les concepts et les identités des
trois phrases d'une paire. Les deux versions, sham/injection, A/B, X/Y et les
doses liées restent ensemble. Avec une seule paire, aucun intervalle n'est
inventé ; le résultat porte la mention `insufficient_text_pairs`. Avec un seul
concept, l'incertitude est explicitement conditionnelle à ce concept.

## Commandes

Depuis la racine du dépôt :

```bash
export PYTHONPATH="$PWD/code"
python -m experiment_5_text_induction.prepare_material_plan --config configs/experiment_5_text_induction/mon_run.yaml
python -m experiment_5_text_induction.run_experiment_5 --config configs/experiment_5_text_induction/mon_run.yaml
python -m experiment_5_text_induction.analyze_experiment_5 --config configs/experiment_5_text_induction/mon_run.yaml
```

Sous PowerShell, la première ligne devient :

```powershell
$env:PYTHONPATH = "$PWD/code"
```

Sur Ruche, les trois étapes sont chaînées avec `afterok` :

```bash
bash jobs/experiment_5_text_induction/submit.sh configs/experiment_5_text_induction/mon_run.yaml
```

## Artefacts

Le plan écrit `pairs.jsonl`, `prompts.jsonl`, `conditions.jsonl`, `trials.jsonl`
et `manifest.json`. Le manifeste contient les hashes des textes, vecteurs,
calibration, configuration et code source des 100 phrases.

Le run écrit les copies exactes du matériel, `trials.jsonl`, `run_manifest.json`
et `complete.json`. Un run interrompu conserve ses lignes et produit
`failure.json`, mais l'analyse refuse de traiter un résultat incomplet.
Les dossiers existants ne sont jamais écrasés.

L'analyse écrit :

- `presence_summary.csv` : métriques de présence par condition et type de texte ;
- `mapping_and_position.csv` : mêmes métriques séparées par A/B et X/Y ;
- `paired_text_effects.csv` : effet principal, effets complémentaires et intervalles ;
- des figures par concept, couche, axe de dose et statistique de calibration ;
- `analysis_manifest.json` : définitions et provenance de l'analyse.

## Vérification

```bash
python -m pytest tests/experiment_5_text_induction \
  tests/experiment_0_calibration -q
```

Les tests couvrent les 16 cellules, les spans exacts, mappings, réponses invalides,
conversion alpha/z, appariement, bootstrap, non-écrasement et un parcours complet
préparation → exécution → analyse sur un petit Llama aléatoire local. Aucun poids
8B n'est téléchargé. Les tests valident le code, pas une conclusion scientifique.
