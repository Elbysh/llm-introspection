# Expérience 4 — dégradation de la classification sémantique

Ce dossier implémente la **section 8 de `cadrage-experiments-new.md`**.

Le modèle classe une phrase selon un concept : par exemple, « cette phrase
exprime-t-elle une trahison ? ». On mesure si une intervention interne modifie
sa réussite, par rapport à la même question sans intervention.

Périmètre retenu après tes précisions : **sham, concept, direction aléatoire fixe,
bruit renouvelé par token**. Le dropout est exclu. L'analyse est descriptive,
sans régression. La comparaison avec l'expérience 1 (étape 7 du document) n'est
pas exécutée : aucun de ses résultats ni de ses fichiers n'est requis.

## Retrouver chaque étape

| Étape de la section 8.5 | Fichier | Ce qu'il fait |
|---|---|---|
| 1. Sélectionner une phrase étiquetée | `step_01_select_examples.py` | Lit les exemples choisis dans `complex_data.json`, conserve classe et provenance, construit les deux mappings X/Y. |
| 2. Vérifier les concepts | `step_02_check_concepts.py` | Refuse un concept évalué égal au concept injecté et une phrase utilisée pour construire la direction injectée. |
| 3. Exécuter le sham | `step_03_run_sham.py` | Forward sans modification, hook actif, réponse et logits de référence. |
| 4. Appliquer l'intervention | `step_04_apply_intervention.py` | Injection à la sortie du bloc, sur les seuls tokens de la phrase ; conversion alpha/z avec les échelles existantes. |
| 5. Extraire réponse et marge | `step_05_score_responses.py` | Réponse, validité, accuracy, logits X/Y, marge correcte et JS si prévue. |
| 6. Répéter les conditions | `step_06_run_conditions.py` | Parcourt les phrases, mappings, familles, couches et doses configurés, avec références sham partagées. |
| Analyse de classification | `analyze_experiment_4.py` | Tableaux et figures des écarts au sham, invalides et JS. |

Les fichiers d'encadrement sont :

- `protocol_config.py` : lecture et validation de la configuration ;
- `prepare_material_plan.py` : préparation des prompts, tokens, directions et essais ;
- `run_experiment_4.py` : vérifications, chargement du modèle, appel des étapes et sauvegarde.

## Ce qui est réutilisé

Les anciens scripts ne sont pas modifiés. L'expérience utilise :

- les exemples positifs et négatifs de `data/dataset/complex_data.json` ;
- les vecteurs et enregistrements `directional_scales.json` de l'expérience 0 ;
- `materialize_direction` pour reconstruire exactement les directions conceptuelles,
  aléatoires fixes et de bruit, y compris la vérification seed/vecteur des aléatoires fixes ;
- `decoder_layer` pour cibler la même sortie de bloc que la calibration ;
- les utilitaires JSONL, SHA-256, commit Git et compatibilité Ruche de l'expérience 0 ;
- le même principe de jobs Slurm : préparation puis exécution avec dépendance `afterok`.

## Données, prompts et réponses

Chaque concept évalué possède autant d'exemples positifs que négatifs. Les indices
choisis sont explicites et commencent à zéro dans chacune des deux listes du JSON.
Une phrase est présentée avec `X = oui, Y = non`, puis avec le mapping inverse.
Le prompt reprend le texte de la section 8.3, sans préremplissage affirmatif.

Seuls les tokens recouvrant le texte de la phrase sont ciblés. Les IDs, offsets
et distances au token de réponse sont enregistrés. Les tokens de bord peuvent
englober un espace ou de la ponctuation : leurs offsets sont conservés.
La question et le concept évalué ne font pas partie du site d'injection.

Les logits sont lus directement au dernier token du préremplissage : ils
prédisent le premier token de réponse. Aucun dernier token n'est rejoué et
aucune génération longue n'est nécessaire.

**Définition de la réponse :** le token de logit maximal dans tout le vocabulaire.
Il est valide uniquement s'il correspond au token X ou Y attendu dans ce prompt.
Un autre token est enregistré comme réponse invalide et compte comme incorrect
dans l'accuracy principale. Le code vérifie que X et Y sont chacun une continuation
d'un seul token ; il ne tronque jamais une réponse multitoken.

La marge est toujours :

```text
logit(bonne réponse) − logit(mauvaise réponse)
```

Une accuracy forcée entre X et Y est également enregistrée, sous le nom
`forced_label_accuracy`. Elle n'est pas confondue avec l'accuracy de la réponse
effective : le modèle peut préférer X à Y tout en répondant autre chose.
Une égalité exacte des deux logits vaut 0,5 pour cette mesure secondaire.

## Paramètres à renseigner

Le gabarit est `configs/experiment_4_task_degradation/template.yaml`.
Il est volontairement incomplet sur les choix que le document ne fixe pas :
concepts évalués, indices des exemples, associations avec les interventions,
doses et sous-ensemble JS. Le programme les exige avant l'exécution au lieu
de choisir des valeurs scientifiques à ta place.

Copie ce fichier sous un nom de run, puis complète `examples`, `conditions`,
`diagnostics.js_example_ids` et les deux dossiers de sortie.

Exemple **de syntaxe**, et non sélection imposée pour l'étude :

```yaml
examples:
  - probed_concept: betrayal
    positive_indices: [0, 1]
    negative_indices: [0, 1]

conditions:
  - condition_id: dust_block03_alpha_example
    family: concept
    intervention_id: Dust
    direction_id: concept__block_03__Dust
    layer: 3
    dose_axis: alpha
    dose: 1.0
    scale_statistic: sd
    probed_concepts: [betrayal]

diagnostics:
  js_example_ids:
    - betrayal__positive__00
    - betrayal__negative__00
```

Chaque ligne de `conditions` décrit une cellule du plan. Ajoute les cellules
correspondant aux couches et doses effectivement retenues. `probed_concepts`
explicite les associations, sans filtrage silencieux d'une association interdite.
`intervention_id` identifie la direction ou la règle de bruit dans les tableaux.

Pour une direction aléatoire fixe, utilise `family: fixed_random` et un identifiant
présent dans ta calibration, par exemple `fixed_random__block_03__0000`.
Pour le bruit, utilise :

```yaml
family: renewed_noise
intervention_id: independent_noise
noise_rule: independent_unit_gaussian_per_trial_token
```

Les autres champs de la condition restent nécessaires ; `direction_id` n'est
pas utilisé pour le bruit. La préparation affecte des directions de bruit déjà
calibrées, par ordre stable de leurs identifiants. Chaque token de chaque essai
reçoit une direction indépendante, y compris entre les mappings et les doses.
Leurs IDs et seeds sont sauvegardés avant le premier essai.

On peut fournir une affectation explicite avec
`noise_assignments: {prompt_id: [direction_id_0, direction_id_1, ...]}`.
Il faut autant de directions que de tokens ciblés et aucune seed partagée entre
réalisations distinctes ou avec la banque aléatoire fixe.

**Si la banque de bruit calibrée est trop petite, la préparation s'arrête.**
Il faut alors préparer et calibrer davantage de directions avec l'expérience 0.
Le code ne recycle pas une direction et ne remplace pas son échelle par une moyenne.

## Calibration et doses

`layer` désigne la sortie de `model.model.layers[layer]`, indexée de 0 à 31.
Les identifiants conceptuels et leurs chemins viennent directement de l'expérience 0 :
aucun nouveau décalage d'indice n'est appliqué aux fichiers de vecteurs.

- `dose_axis: alpha` : coefficient demandé égal à `dose` ;
- `dose_axis: z` : coefficient demandé égal à `dose × scale` ;
- `scale_statistic: sd` : écart-type calibré ;
- `scale_statistic: mad_corrected` : MAD corrigée existante, si cette sensibilité est souhaitée.

Pour le bruit, chaque token utilise l'échelle de **sa propre direction**.
Les amplitudes réalisées sont mesurées après l'addition dans le dtype du modèle :
elles peuvent différer de la dose demandée à cause des arrondis.
Les hooks sont toujours retirés, y compris en cas d'erreur.

Le mode `development` permet de vérifier le parcours avec la calibration actuelle.
Le mode `frozen`, comme dans l'expérience 0, exige notamment des révisions immuables,
une calibration figée, les 32 couches, les trois familles retenues et les deux axes.
La configuration disponible du dépôt reste une calibration de développement.

## Commandes

Depuis la racine, avec l'environnement Python du projet activé :

```bash
export PYTHONPATH="$PWD/code"
python -m experiment_4_task_degradation.prepare_material_plan --config configs/experiment_4_task_degradation/mon_run.yaml
python -m experiment_4_task_degradation.run_experiment_4 --config configs/experiment_4_task_degradation/mon_run.yaml
python -m experiment_4_task_degradation.analyze_experiment_4 --run results/experiment_4_task_degradation/mon_run --output results/experiment_4_task_degradation/mon_analyse
```

Sous PowerShell, la première ligne devient `$env:PYTHONPATH = "$PWD/code"`.
`mon_run.yaml` et les noms de sorties ci-dessus sont à remplacer par ta configuration.

Sur Ruche :

```bash
bash jobs/experiment_4_task_degradation/submit.sh configs/experiment_4_task_degradation/mon_run.yaml
```

## Résultats et lecture

Le plan contient les prompts complets, `conditions.jsonl`, `trials.jsonl` et
`manifest.json`. Ce dernier compte à l'avance les essais, sham et mesures JS.

Le run conserve :

- `prompts.json` et `conditions.json` : textes et conditions exacts ;
- `trials.jsonl` : une ligne par sham ou intervention, avec référence au sham,
  réponse, classe correcte, mapping, logits, marge, amplitudes par token et JS éventuelle ;
- `run_manifest.json` : configuration, hashes des entrées, versions, révisions et commit ;
- `complete.json` : nombre final de lignes et hash des essais.

Pour une phrase/mapping/couche, le sham est mesuré une fois puis partagé entre
les conditions identiques sans injection. Tous les écarts sont calculés contre
ce même sham : aucune moyenne globale de référence ne le remplace.

L'analyse produit :

- `classification_summary.csv` : accuracy, accuracy sham, différence d'accuracy,
  marge, différence de marge, invalides, JS moyenne et nombre de cas JS ;
- `classification_by_concept_and_mapping.csv` : les mêmes mesures séparées
  selon le concept évalué et le mapping, pour rendre leurs biais visibles ;
- des PNG par couche, axe et statistique de calibration ;
- `analysis_manifest.json` : provenance et périmètre de l'analyse.

`delta_accuracy = accuracy_intervention − accuracy_sham` : une valeur négative
signifie une baisse. `delta_correct_margin` suit le même sens. La JS est calculée
sur tout le vocabulaire du prochain token, en nats, uniquement pour les exemples
présélectionnés ; hors sous-ensemble, sa valeur est `null`, jamais zéro par défaut.
Une JS élevée indique un changement de sortie, pas nécessairement une baisse de réussite.

Les courbes ne mélangent pas alpha/z, SD/MAD, couches ou sélections d'exemples
différentes. Les tableaux restent descriptifs : ils ne concluent pas à une
significativité ni à une relation avec la détection, qui n'est pas mesurée ici.

Les résultats ne sont pas écrasés. Un échec numérique arrête le run et produit
`failure.json`, sans supprimer silencieusement une dose. Les lignes déjà calculées
restent inspectables ; l'analyse refuse un fichier incomplet ou modifié.
La reprise automatique n'est pas implémentée.

## Tests

```bash
python -m pytest tests/experiment_4_task_degradation tests/experiment_0_calibration -q
```

Les tests couvrent les mappings, spans, exclusions conceptuelles, coefficients
alpha/z, bruit indépendant par token, norme réalisée après arrondi, réponses
invalides, JS, nettoyage des hooks et un parcours complet préparation → run →
analyse sur un petit Llama aléatoire local. Aucun poids 8B n'est téléchargé pour
ces tests. Ils vérifient le code, pas un résultat scientifique du modèle 8B.
