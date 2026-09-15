# `s(ℓ,v)` recalculée dans le contexte comportemental, et l'expérience 1 refaite en `z`

**Résultats.** Note technique, 14 septembre 2026.
Fait suite à `calibration-contexte-probleme-et-correctif.md`, dont elle exécute les
points 1, 3 et 5 du plan d'action.

---

## Résumé

`s(ℓ,v)` a été réestimée sur le prompt 2AFC exact que l'expérience 1 perturbe, et non
plus sur des phrases isolées. Les deux critères d'acceptation du §8.6 de la note de
diagnostic sont vérifiés. Le rapport d'échelle concept/aléatoire tombe de **29–122 à
1,5–3,0**, et les estimateurs SD et MAD s'accordent enfin.

Le balayage en `z` a été refait sur les 32 blocs avec une grille reconstruite,
`z ∈ [0,0625 ; 8192]`, 18 doses. La règle de vérification du §8.5 est satisfaite sur
l'amplitude **réellement délivrée** : aucune des 128 cellules (bloc, famille) ne reste
sous le seuil mesuré sur la grille α.

Deux conclusions.

**L'effet catégoriel disparaît.** Les quatre familles répondent maintenant, avec des
transitions au même ordre de dose. L'avantage conceptuel subsiste mais il est
**modeste** : rapport 1,18 en moyenne sur la bande vivante, contre la séparation
apparemment catégorielle du balayage précédent.

**La coupure de profondeur au bloc 14 est confirmée indépendamment.** Elle n'était
jusqu'ici établie qu'à `α` apparié. Elle se reproduit à `z` apparié, avec une grille
qui délivre 400 à 21 000 d'amplitude brute aux blocs profonds : le zéro y est un vrai
zéro, pas un défaut d'étendue.

---

## 1. Ce qui a été fait

| | |
|---|---|
| Manifeste 2AFC | `data/experiment_0_calibration/contexts_2afc_llama.jsonl`, produit par `code/experiments/build_2afc_calibration_manifest.py` |
| Configuration | `configs/experiment_0_calibration/llama_2afc_full.yaml` |
| Calibration | `results/experiment_0_calibration_2afc` (job 8244, 25 min, 32 blocs, 79 488 directions) |
| Balayage `z` | `results/experiment1/z2afc_all` (jobs 8255–8258, 4 groupes de 8 blocs, 253 460 essais) |

Le manifeste suit le §8.3 : les 100 phrases du corpus appariées en 50 paires de
longueurs voisines, chaque paire rendue dans les 2 ordres physiques × 2 ordres
d'étiquettes. **200 prompts, 2464 positions**, chaque phrase présentée quatre fois et
deux fois à chaque emplacement. Le plan est donc équilibré en emplacement et en
étiquette, ce qui justifie une échelle unique par `(bloc, direction)`.

Tout le reste est tenu fixe contre `development_full.yaml` — même corpus, mêmes
couches, mêmes concepts, mêmes graines de directions — de sorte que la seule quantité
qui change entre les deux calibrations est le contexte de présentation.

### 1.1 Un point que la note de diagnostic n'avait pas anticipé

Le §8.2 annonçait qu'aucune modification du moteur n'était nécessaire. Ce n'est pas
tout à fait exact. Dans le prompt, **la phrase n'est pas alignée sur les tokens** : le
premier token porte aussi l'espace de son étiquette `A) ` (`ĠShe` couvre les
caractères 314–318 pour une borne à 315) et le dernier porte le saut de ligne qui
suit. Deux tokens à cheval par cible, quatre par prompt.

La politique `all_sentence_tokens` refuse exactement ces tokens-là, et la construction
du plan échouait sur le premier contexte. Élargir les bornes en caractères n'est pas
une option : la vérification `rendered[char_start:char_end] != sentence` rejetterait
alors la ligne. Les jeter non plus : `build_localization_prompt` retient **tous les
tokens qui chevauchent** la phrase, donc l'expérience 1 les perturbe, et le cadrage
§3.7 dit « tous les tokens de la phrase ciblée ». Les écarter aurait calibré sur 1664
des 2464 positions, en excluant systématiquement le premier et le dernier token de
chaque phrase.

Le correctif est une seconde politique de positions,
`all_overlapping_sentence_tokens` (`POSITION_POLICIES` dans `protocol_config.py`).
`all_sentence_tokens` conserve son comportement exact, donc la calibration isolée
publiée n'est pas affectée.

**Vérifié** : les 2464 positions admissibles du plan **égalent exactement**, cible par
cible, les `token_range` que `build_localization_prompt` perturbe. Elles sont aux
positions 57 à 75 du prompt, et **aucune n'est en position 0**.

---

## 2. La calibration : les deux critères du §8.6

`code/analysis/check_calibration_context.py`, blocs 1, 5, 9 et 13.

| | contexte isolé | prompt 2AFC |
|---|---|---|
| `s_SD/s_MAD`, toutes familles | 15 à 2 360 | **0,96 à 1,19** |
| masse de queue × tokens/séquence | 0,94 | **≤ 0,48** |
| plancher gaussien de cette dernière | 0,11 | 0,11 |

Le premier critère est celui qui tranche. En contexte isolé, `s_SD/s_MAD` valait 2 360
pour le concept au bloc 1 ; il vaut **0,957**. La distribution des projections dans le
contexte que l'expérience perturbe réellement est essentiellement gaussienne, pour les
trois familles.

Le second critère demande une précaution de lecture. La masse de queue
`f = |moyenne − médiane| / |extrême − médiane|` ne s'annule pas sur un échantillon
gaussien fini : `moyenne − médiane` fluctue comme `σ/√n` tandis que `p99 − médiane`
vaut environ `2,33 σ`, donc `f ≈ 0,43/√n`, soit 0,0087 à `n = 2464`. Les valeurs
mesurées, 0,013 à 0,039, sont de cet ordre. Le résidu est du bruit d'échantillonnage,
pas une queue. En contexte isolé `f` valait 0,15, soit dix-sept fois le plancher.

Le dénominateur du produit est le nombre de **séquences rendues**, non de phrases
distinctes : le manifeste présente chaque phrase quatre fois, et diviser les positions
par 100 gonflerait le produit d'un facteur quatre pour une raison étrangère aux
queues.

### 2.1 Ce que devient le rapport d'échelle entre familles

C'est la quantité que la division par `s` est censée éliminer.

| bloc | isolé, par SD | isolé, par MAD | prompt, par SD | prompt, par MAD |
|---:|---:|---:|---:|---:|
| 1 | 122 | 2,69 | **1,46** | 1,62 |
| 5 | 56,3 | 1,81 | **2,12** | 1,85 |
| 9 | 28,8 | 2,53 | **2,24** | 2,24 |
| 13 | 34,4 | 2,87 | **3,00** | 2,63 |

Les deux estimateurs s'accordent désormais à 10 % près. Le facteur 29–122 mesuré sur
la calibration isolée était entièrement l'effet du token de position 0, et la MAD
approximait accidentellement la bonne quantité, comme l'annonçait le §9.

### 2.2 `s(ℓ,v)` dépend maintenant de la profondeur, non de la famille

| bloc | 0 | 3 | 9 | 13 | 16 | 24 | 31 |
|---|---:|---:|---:|---:|---:|---:|---:|
| concept | 0,0302 | 0,0733 | 0,1898 | 0,2770 | 0,3234 | 0,4868 | 2,843 |
| aléatoire | 0,0142 | 0,0395 | 0,0846 | 0,0924 | 0,1266 | 0,2551 | 0,7505 |
| bruit | 0,0145 | 0,0401 | 0,0800 | 0,0970 | 0,1260 | 0,2615 | 0,6990 |

L'échelle croît régulièrement d'un facteur 200 du bloc 0 au bloc 31. C'est la
profondeur, et non plus la famille, qui gouverne la grille.

La provenance est propre : le sha256 de la configuration sur disque égale celui
enregistré dans `run_manifest.json`, donc l'expérience 1 tourne **sans**
`--allow_calibration_mismatch`, contrairement à l'ancienne calibration Llama.

---

## 3. La grille `z` reconstruite

`code/analysis/plan_z_grid.py` applique la règle du §8.5 : la dose maximale doit
délivrer, pour **chaque** famille à **chaque** bloc, plus d'amplitude que le seuil de
cette famille sur la grille α.

Deux contraintes la fixent :

- `z_max ≥ 6372` — le bloc 1 aléatoire, que la grille α n'a jamais amené à 75 %. Sa
  courbe plate ne veut donc rien dire tant que `z` ne lui délivre pas au moins le haut
  de la grille α, soit 128, contre `s = 0,0201`.
- `z_min ≤ 0,088` — le bloc 31 concept, `s = 2,843`, pour atteindre le plancher 0,25 de
  la grille α.

D'où **`z ∈ [0,0625 ; 8192]`, 18 doses au pas ×2**, et zéro cellule non couverte. Pour
mémoire, la grille du balayage `full32_all` s'arrêtait à 20,48, ce qui avec ces
échelles plafonne vers `α = 0,4` au bloc 1 : elle ne mesurerait rien.

C'est le bénéfice que le §8.5 annonçait : le rapport d'échelle entre familles étant
tombé à 1,5–3,0, **une grille unique partagée par les quatre familles et les 32 blocs**
redevient possible. Elle paie son étendue aux deux bouts — `z = 8192` au bloc 31
demande `α ≈ 23 000` — mais aucune dose n'a produit d'activation non finie.

### 3.1 Vérification sur l'amplitude délivrée

Contrôle a posteriori sur `trials.csv`, et non sur la dose demandée.

| bloc | famille | α délivrée à `z_max` | seuil α₇₅ | marge | acc. max |
|---:|---|---:|---:|---:|---:|
| 1 | concept | 214 | 4 | 53× | 0,89 |
| 1 | aléatoire | 179 | > 128 | 1,4× | 0,72 |
| 9 | concept | 1612 | 8 | 202× | 0,85 |
| 9 | dropout | 283 | 8 | 35× | 0,82 |
| 13 | aléatoire | 719 | > 128 | 5,6× | 0,70 |
| 16 | concept | 2643 | > 128 | 21× | 0,50 |
| 31 | concept | 21 100 | > 128 | 165× | 0,50 |

**Sur les 128 cellules (bloc, famille), aucune ne reste sous sa cible**, et 100 % des
activations restent finies, y compris aux doses extrêmes.

---

## 4. Ce que dit le balayage en `z` recalibré

### 4.1 Les quatre familles répondent

Accuracy ajustée par dose, bloc 9 :

| famille | z=8 | 16 | 32 | 64 | 128 | 256 | 1024 | 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| concept | 0,60 | 0,72 | 0,79 | 0,75 | 0,75 | 0,77 | 0,83 | 0,85 |
| aléatoire | 0,55 | 0,63 | 0,71 | 0,78 | 0,82 | 0,76 | 0,62 | 0,55 |
| bruit | 0,49 | 0,47 | 0,60 | 0,79 | 0,85 | 0,84 | 0,62 | 0,65 |
| dropout | 0,49 | 0,52 | 0,64 | 0,72 | 0,82 | 0,79 | 0,59 | 0,76 |

Chaque famille a une transition **à l'intérieur** de la grille, et toutes la
franchissent au même ordre de dose, `z ≈ 32` à `256`, puis redescendent. C'est
exactement ce que la quasi-égalité des échelles prédit. Les courbes témoins plates du
balayage précédent étaient l'artefact d'étendue, et rien d'autre.

La non-monotonie aux fortes doses reproduit celle déjà observée à `α` apparié ; elle
n'est donc pas un effet de la paramétrisation.

### 4.2 L'avantage conceptuel est modeste

Contraste de localisation apparié `S`, poolé sur les doses.

Ordonnancement global, les 32 blocs poolés :

| famille | essais appariés | `S` moyen | t |
|---|---:|---:|---:|
| concept | 46 080 | **+0,1185** | 77,4 |
| dropout | 23 040 | +0,0936 | 46,4 |
| bruit | 23 040 | +0,0926 | 49,7 |
| aléatoire | 34 560 | +0,0727 | 51,4 |

L'ordonnancement `concept > bruit ≈ dropout > aléatoire` du cadrage tient, mais le
rapport concept/aléatoire vaut **1,63**, non un ordre de grandeur. Null de permutation
à 200 tirages : `S` observé +0,0968 contre +0,0002 ± 0,0010, soit 100 écarts-types.

Par bloc, sur la bande vivante, rapport du concept au **meilleur** témoin :

| bloc | 0 | 3 | 7 | 9 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rapport | 1,21 | 0,98 | 1,20 | 1,16 | 1,47 | 1,66 | **1,90** |

Il est proche de 1 aux blocs précoces — et passe sous 1 aux blocs 1 à 5, où le
dropout ou l'aléatoire dépassent le concept — puis croît régulièrement jusqu'à 1,90 au
bloc 13. Moyenné sur les blocs 0 à 15 : concept +0,234 contre +0,199 pour le meilleur
témoin, **rapport 1,18**.

**Lecture.** À `z` apparié dans le bon contexte, le concept ne se détache nettement
qu'à l'approche de la coupure de profondeur. Aux blocs précoces il fait ce que font
les directions génériques. Cela reste à rapprocher du §4 de la note de diagnostic :
aux blocs précoces, les vecteurs conceptuels sont quasi colinéaires à la direction de
puits, donc injecter un « concept » y revient surtout à injecter cette direction.
**Ce chantier n'est pas traité ici et la recalibration ne le corrige pas.**

### 4.3 La coupure de profondeur au bloc 14 est confirmée

| bloc | 12 | 13 | 14 | 15 | 16 | 24 | 31 |
|---|---:|---:|---:|---:|---:|---:|---:|
| concept | +0,273 | +0,235 | +0,007 | +0,007 | +0,001 | +0,002 | 0,000 |
| aléatoire | +0,134 | +0,123 | −0,015 | −0,005 | −0,007 | −0,002 | 0,000 |

La chute est nette d'un bloc à l'autre, pour les quatre familles. Elle n'était
jusqu'ici établie qu'à `α` apparié ; elle se reproduit ici sous une paramétrisation
indépendante, et avec une grille qui délivre 400 à 21 000 d'amplitude brute au-delà du
bloc 14. Le zéro profond est un zéro, pas un défaut de couverture.

Le bloc 31 donne exactement le hasard pour toutes les familles et toutes les doses :
aucune couche d'attention ne le suit, donc une perturbation aux positions des phrases
ne peut pas atteindre le token de réponse. C'est un contrôle négatif de structure.

---

## 5. Statut des résultats

| | statut |
|---|---|
| Résultats en `z` de `full32_all` | **Remplacés** par `z2afc_all`. |
| Résultats en `α` de `full32_all` | **Intacts** — ils n'appellent aucune échelle calibrée. |
| Ordonnancement des familles | **Tient**, à `z` apparié comme à `α`. |
| Amplitude de l'avantage conceptuel | **Établie à 1,18–1,63**, et non catégorielle. |
| Coupure de profondeur au bloc 14 | **Confirmée** sous les deux paramétrisations. |
| Seuils `z₇₅` | Toujours sans objet : les courbes restent non monotones. |
| Colinéarité des vecteurs conceptuels avec la direction de puits | **Non traitée**, et non corrigée par la recalibration. Tant qu'elle ne l'est pas, l'écart concept/témoins ne peut pas être attribué au contenu conceptuel. |

## 6. Reproduire

```bash
python code/experiments/build_2afc_calibration_manifest.py
python -m experiment_0_calibration.prepare_material_plan \
    --config configs/experiment_0_calibration/llama_2afc_full.yaml
sbatch --partition=prod80 --gres=gpu:nvidia_a100-sxm4-80gb:1 --mem=120G --time=06:00:00 \
    jobs/experiment_0_calibration/03_run_experiment_0.sbatch \
    configs/experiment_0_calibration/llama_2afc_full.yaml

python code/analysis/check_calibration_context.py \
    --calibration_dir results/experiment_0_calibration_2afc \
    --baseline_dir results/experiment_0_calibration
python code/analysis/plan_z_grid.py \
    --calibration_dir results/experiment_0_calibration_2afc \
    --alpha_trials results/experiment1/full32_all/trials.csv

for g in 0 1 2 3; do
  LAYERS="..." sbatch --partition=prod80 --gres=gpu:nvidia_a100-sxm4-80gb:1 \
      --mem=120G --time=03:00:00 jobs/experiment1_z_recalibrated.sbatch \
      configs/experiment_0_calibration/llama_2afc_full.yaml "z2afc-g${g}"
done
python code/analysis/merge_experiment1_runs.py \
    --run_dir results/experiment1/z2afc-g0 --run_dir results/experiment1/z2afc-g1 \
    --run_dir results/experiment1/z2afc-g2 --run_dir results/experiment1/z2afc-g3 \
    --out results/experiment1/z2afc_all
python code/analysis/experiment1_localization_report.py \
    --run_dir results/experiment1/z2afc_all
```
