# Expérience 0 — résultats de calibration

Ce dossier contient les résultats du calcul de l’échelle directionnelle naturelle
`s(l, v)` sur le corpus témoin. Il s’agit d’une calibration de développement :
elle ne calcule encore aucune dose `alpha` ou `z` et ne contient aucun résultat
2AFC.

> **Validation du nouveau calcul :** les 20 032 graines stochastiques sont
> uniques et aucun chevauchement ne subsiste entre `fixed_random` et
> `renewed_noise`. Les 320 directions aléatoires fixes réutilisent la banque
> persistée du dépôt ; les 19 712 directions de bruit ont été recalculées avec
> leur nouvel espace de graines.

## Périmètre de ce calcul

- modèle : `meta-llama/Llama-3.1-8B-Instruct` ;
- 100 phrases témoins ;
- 616 positions de tokens admissibles ;
- 32 sorties de blocs décodeur ;
- 20 352 couples couche-direction au total.

Pour chaque couche, les 636 directions sont réparties ainsi :

- 10 directions de concept ;
- 10 directions aléatoires fixes ;
- 616 directions de bruit renouvelé, soit une par position admissible.

Sur les 32 couches, cela représente 320 directions de concept, 320 directions
aléatoires fixes et 19 712 directions de bruit renouvelé.

## Fichiers

### `directional_scales.csv`

Table principale, facile à ouvrir dans Python, R ou un tableur. Une ligne
correspond à un couple `(couche l, direction v)`. Elle contient notamment :

- l’identité et la famille de la direction ;
- la couche et le site d’activation ;
- le nombre de phrases et de positions utilisées ;
- la moyenne, la médiane, la SD et la MAD corrigée des projections naturelles
  `<h, v>` ;
- les quantiles `p01`, `p05`, `p95` et `p99` ;
- les intervalles et coefficients de variation obtenus par bootstrap de phrases ;
- les graines, provenances et métadonnées utiles à la reproductibilité ;
- `valid_for_sd_normalization`, qui indique si la valeur peut servir de
  calibration SD ultérieure.

### `directional_scales.json`

Même table et mêmes 20 352 enregistrements que le CSV, dans un format qui
préserve plus directement les types et métadonnées pour les scripts.

### `projections.npz`

Archive NumPy compressée des projections naturelles brutes. Chaque entrée est
indexée par `direction_id` et contient les 616 valeurs `<h, v>` observées aux
positions admissibles. Ce fichier permet de recalculer les statistiques et les
figures sans refaire une passe du modèle.

### `run_manifest.json`

Carte d’identité du calcul : protocole, statut de développement, commit Git,
révision résolue du modèle, versions logicielles, tailles des données et hashes
des fichiers de préparation. Il permet de vérifier précisément avec quel plan
et quel modèle les résultats ont été produits.

> **Note de provenance :** le champ `git_commit` du manifeste décrit le HEAD du
> checkout présent sur Ruche, pas exactement les sources exécutées, car les
> fichiers de cette branche ont été transférés directement par SCP. Le hash de
> configuration enregistré dans le manifeste correspond bien au fichier local
> utilisé (`3545c6bda6bd4f0d3a08882dd073a85a180e2c4d0df8aa357b953ab63977ddf9`).
> Il faut donc utiliser ce hash, les artefacts versionnés et le code de cette
> branche pour retracer ce run, plutôt que le seul champ `git_commit`.

## Figures

Le dossier `figures/` contient :

- `experiment_0_scales_sd.png` : `s_SD(l,v)` selon la couche, en échelle
  linéaire et dans trois panneaux séparés. Les dix concepts sont tracés
  individuellement. Les contrôles aléatoires fixes et le bruit renouvelé sont
  résumés par leur médiane, leur IQR et leurs quantiles 5–95 %.
- `experiment_0_scales_sd_log.png` : même figure avec une échelle verticale
  logarithmique dans chacun des trois panneaux.
- `experiment_0_sd_vs_mad.png` : comparaison entre SD et MAD corrigée dans un
  panneau indépendant par famille. Chaque panneau possède ses propres limites
  logarithmiques afin que les 19 712 bruits ne masquent pas les autres points.
  La diagonale correspond à `SD = MAD`.
- `experiment_0_sd_over_mad.png` : médiane du rapport `SD / MAD corrigée` par
  couche et famille, accompagnée de son IQR entre directions. Un rapport élevé
  signale une sensibilité plus forte de la SD aux queues de distribution ou aux
  valeurs extrêmes.
- `experiment_0_bootstrap_stability.png` : coefficient de variation bootstrap
  médian de la SD par couche et famille, avec l’IQR entre directions. Plus il
  est faible, plus l’estimation est stable vis-à-vis du choix des phrases.
- `experiment_0_projection_distributions.png` : distributions centrées-réduites
  aux blocs 1, 16 et 30, comparées à une loi normale standard. Les dix directions
  conceptuelles et aléatoires fixes sont utilisées. Pour le bruit renouvelé,
  25 identifiants régulièrement espacés dans la liste triée sont utilisés, et
  non les 25 premiers. Chaque direction est standardisée avant agrégation ; ce
  graphique compare donc la forme des distributions, pas leur amplitude.
- `experiment_0_concept_scale_heatmap.png` : heatmap `concept × couche` de
  `log10(s_SD)`, destinée à rendre visibles les différences entre concepts que
  masquerait une moyenne de famille.
- `experiment_0_relative_scales.png` : en haut, rapport entre l’échelle de chaque
  concept et la médiane des directions aléatoires fixes de la même couche ; en
  bas, rapport entre le bruit renouvelé et ce même contrôle aléatoire. La ligne
  `1` représente l’égalité avec le contrôle.

## Statut scientifique

Ces sorties portent le statut `development`. Elles servent à examiner la
calibration et à arrêter les choix du protocole avant de figer les échelles.
Elles ne doivent pas être interprétées comme les performances finales de la
tâche 2AFC.

## Vérifications effectuées avant versionnement

- 20 352 lignes dans le CSV et le JSON, avec les mêmes `direction_id` ;
- 320 directions de concept, 320 directions aléatoires fixes et 19 712
  directions de bruit renouvelé ;
- 32 couches contenant chacune exactement 10 concepts, 10 directions
  aléatoires fixes et 616 directions de bruit ;
- 20 032 graines stochastiques toutes distinctes, sans intersection entre les
  deux familles de contrôle ;
- 20 352 tableaux de projections de forme `(616,)`, tous finis ;
- les 20 352 calibrations sont marquées `valid_for_sd_normalization` ;
- les huit figures attendues ont été générées.
