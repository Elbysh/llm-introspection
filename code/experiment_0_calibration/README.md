# Expérience 0 — calibration de l’échelle naturelle

Ce dossier contient uniquement l’expérience 0 du protocole : estimer
`s(couche, direction)` à partir d’activations naturelles, sans intervention et
sans réponse comportementale.

Il ne contient ni tâche 2AFC, ni injection, ni grille de doses. La conversion
`alpha = z × s(couche, direction)` est volontairement reportée.

## Question mesurée

À chaque position de token admissible, pour une sortie de bloc Transformer
`h` et une direction unitaire `v`, l’expérience calcule :

```text
p(contexte, token, couche, direction) = <h, v>
```

L’échelle naturelle est ensuite estimée avec :

```text
s_SD  = écart-type échantillonnal des projections
s_MAD = 1.4826 × médiane(|projection - médiane(projection)|)
```

La SD est l’analyse principale. La MAD corrigée est l’analyse de sensibilité.

## Organisation du dossier

```text
experiment_0_calibration/
├── README.md
├── __init__.py
│
├── protocol_config.py
│   └── charge et valide tous les choix du protocole
│
├── prepare_concept_vectors.py
│   └── prérequis : crée les vecteurs conceptuels manquants
├── prepare_material.py
│   └── construit contextes, positions et directions planifiées
├── prepare_material_plan.py
│   └── commande qui écrit le plan expérimental sur disque
│
├── step_01_02_collect_natural_activations.py
│   ├── étape 1 : forward naturel sans intervention
│   └── étape 2 : extraction de h aux positions admissibles
├── step_03_project_activations.py
│   └── étape 3 : produit scalaire <h, v>
├── step_04_estimate_scales.py
│   └── étape 4 : moyenne, SD, médiane, MAD et quantiles
├── step_05_plot_distributions.py
│   └── étape 5 : distributions et figures diagnostiques
├── step_06_bootstrap_stability.py
│   └── étape 6 : bootstrap par phrase et intervalles
├── step_07_persist_and_freeze.py
│   └── étape 7 : validation, traçabilité et écriture sans écrasement
│
└── run_experiment_0.py
    └── orchestrateur : appelle les étapes, sans réimplémenter leur logique
```

Les étapes 1 et 2 partagent un fichier parce que les activations doivent être
extraites pendant que les hooks du forward sont actifs. Les deux opérations
restent séparées en fonctions identifiables.

## Préparation du matériel

La préparation précède l’étape 1. Elle fixe ce qui sera mesuré et écrit quatre
fichiers dans le `plan_dir` de la configuration.

### `contexts.jsonl`

Une ligne par présentation au modèle :

- texte complet réellement tokenisé ;
- identifiant du contexte ;
- phrases présentes dans ce contexte ;
- spans de caractères ;
- IDs de tokens ;
- positions admissibles.

### `observations.jsonl`

Une ligne par position effectivement calibrée :

- `observation_id` ;
- `sentence_id` et hash de la phrase ;
- `context_id` ;
- `token_index` dans le contexte ;
- `sentence_token_index` dans la phrase ;
- ID, texte et span du token.

Cet index empêche d’aplatir anonymement les tokens et permet le bootstrap par
phrase.

### `directions.jsonl`

Une ligne par direction et couche avec famille, identifiant, graine et
provenance.

La configuration de développement contient :

| Famille | Par couche | Total pour 32 couches |
| --- | ---: | ---: |
| concept | 10 | 320 |
| fixed_random | 10 | 320 |
| renewed_noise | 1 par position admissible | `32 × nombre de positions` |

Avec les 616 positions observées lors du précédent passage, cela représente
19 712 directions de bruit et 20 352 directions au total. Le nombre exact est
écrit après tokenisation dans le manifeste, il n’est pas supposé à l’avance.

Les familles ont des rôles différents :

- `concept` charge une direction conceptuelle existante et la normalise ;
- `fixed_random` utilise la banque persistée
  `data/saved_vectors/llama/random_s{sample}_{bloc}_avg.pt`. Chaque fichier est
  hashé dans le plan et vérifié bit à bit contre sa seed lors du calcul ;
- `renewed_noise` tire une direction indépendante pour chaque
  `(bloc, position, répétition)` et enregistre la position à laquelle elle sera
  associée.

Les deux familles stochastiques utilisent des intervalles de seeds séparés. La
banque `fixed_random` conserve sa convention historique
`2026091001 + bloc × 100000 + sample`, tandis que `renewed_noise` utilise un
espace indépendant. La préparation et l’exécution refusent un plan contenant
la moindre seed dupliquée : deux contrôles supposés indépendants ne peuvent donc
pas reconstruire accidentellement le même vecteur.

### `manifest.json`

Résumé vérifiable : expérience, version du protocole, configuration, modèle,
tokenizer, versions logicielles, contexte, politique de positions et nombres
de phrases, contextes, observations et directions.

## Correspondance exacte avec le déroulé

### Étape 1 — passer chaque phrase sans intervention

`collect_natural_activations` installe un hook sur la sortie des 32 blocs puis
effectue les forwards avec `use_cache=False`. Aucune activation n’est modifiée.

### Étape 2 — extraire les activations admissibles

`select_admissible_activations` utilise uniquement les positions de
`observations.jsonl`. Il vérifie aussi que la tokenisation n’a pas changé entre
la préparation et le calcul.

### Étape 3 — projeter sur chaque direction

`project_activations` construit des matrices de directions par blocs et
calcule :

```text
projections = activations @ directions.T
```

Les directions sont traitées par morceaux pour limiter la mémoire GPU.

### Étape 4 — estimer les échelles

`estimate_point_scales` calcule séparément, pour chaque couple
`(decoder_block_index, direction_id)` :

- moyenne ;
- SD avec `ddof=1` ;
- médiane ;
- MAD corrigée ;
- quantiles 1 %, 5 %, 95 % et 99 %.

La règle ponctuelle est `equal_token` : chaque position admissible a le même
poids. Elle est explicite dans la configuration et les résultats.

### Étape 5 — tracer les distributions

`step_05_plot_distributions.py` produit :

- échelles SD par bloc et par famille, en version linéaire et logarithmique ;
- SD contre MAD corrigée, dans un panneau indépendant par famille ;
- rapport SD/MAD médian et IQR par famille et bloc ;
- stabilité bootstrap médiane et IQR ;
- distributions standardisées aux blocs early, middle et late, avec référence
  gaussienne et sélection déterministe des directions ;
- heatmap des échelles conceptuelles par concept et couche ;
- rapports concept/random et bruit/random par couche.

Le rendu est exécuté après l’écriture des données afin que les figures soient
toujours reproductibles depuis les artefacts sauvegardés.

### Étape 6 — vérifier la stabilité par bootstrap

Le bootstrap rééchantillonne les 100 phrases, pas les tokens indépendamment.
Quand une phrase est tirée, tous ses tokens et toutes ses présentations sont
conservés ensemble. Les mêmes tirages sont utilisés pour toutes les directions
et couches.

Les sorties incluent :

- intervalles bootstrap SD et MAD ;
- coefficient de variation bootstrap SD et MAD ;
- nombre de réplications et niveau de confiance.

### Étape 7 — figer l’échelle

`step_07_persist_and_freeze.py` vérifie le plan avant de charger le modèle et
écrit ensuite :

- `directional_scales.json` ;
- `directional_scales.csv` ;
- `projections.npz` ;
- `run_manifest.json` ;
- dossier `figures/`.

Il enregistre les hashes du plan, le commit Git, les révisions du modèle et du
tokenizer et les versions logicielles. Il refuse toujours d’écraser un dossier
de résultats existant.

## Indexation des couches

`decoder_block_index` désigne toujours le bloc hooké :

```text
model.model.layers[decoder_block_index]
```

Hugging Face place les embeddings dans `hidden_states[0]`. La direction
conceptuelle correspondant à la sortie du bloc `l` se trouve donc dans
`hidden_states[l + 1]`. Par exemple :

| Sortie calibrée | Fichier conceptuel |
| ---: | ---: |
| bloc 0 | hidden state 1 |
| bloc 16 | hidden state 17 |
| bloc 31 | hidden state 32 |

`prepare_concept_vectors.py` crée le hidden state 32 s’il manque, sans modifier
les utilitaires historiques du dépôt.

## Configuration actuelle : développement, pas calibration figée

La configuration est :

```text
configs/experiment_0_calibration/development_full.yaml
```

Elle utilise actuellement les phrases isolées et une réalisation de bruit par
position. Elle est explicitement marquée `development`.

Avant de passer à `frozen`, il faut encore fixer dans le protocole :

1. le contexte comportemental exact ;
2. le nombre final de répétitions de bruit par position ;
3. les hashes immuables du modèle et du tokenizer ;
4. le split conceptuel `development`/`hold_out`.

Pour un contexte final contenant des paires ou plusieurs occurrences, le mode
`external_manifest` accepte des lignes :

```json
{
  "context_id": "identifiant_unique",
  "rendered_text": "prompt complet exact",
  "targets": [
    {"sentence_id": "localization_000", "char_start": 120, "char_end": 174}
  ]
}
```

Le validateur interdit de déclarer `frozen` une configuration utilisant encore
`main`, un contexte de développement ou des splits non assignés.

## Exécution sur Ruche

La chaîne de jobs comporte trois étapes techniques :

```text
01_prepare_concept_vectors.sbatch
  └── 02_prepare_material_plan.sbatch
        └── 03_run_experiment_0.sbatch
```

Soumission :

```bash
bash jobs/experiment_0_calibration/submit.sh development_full
```

Commande équivalente par étape :

```bash
python -m experiment_0_calibration.prepare_concept_vectors \
  --config configs/experiment_0_calibration/development_full.yaml

python -m experiment_0_calibration.prepare_material_plan \
  --config configs/experiment_0_calibration/development_full.yaml

python -m experiment_0_calibration.run_experiment_0 \
  --config configs/experiment_0_calibration/development_full.yaml
```
