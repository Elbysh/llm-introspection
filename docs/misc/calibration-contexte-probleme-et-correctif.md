# L'échelle naturelle `s(ℓ,v)` est mesurée dans le mauvais contexte

**Problème, preuve et correctif.** Note technique, 13 septembre 2026.
Concerne l'expérience 0 (calibration) et tous les résultats en `z` de l'expérience 1.

---

## Résumé

L'expérience 0 calibre `s(ℓ,v)` sur des phrases rendues **isolément**. Dans ce
contexte, le premier token de chaque phrase occupe la position 0 de sa propre séquence
— la position de puits d'attention, où Llama loge une activation de norme démesurée.
La politique `all_sentence_tokens` l'inclut dans l'échantillon, où il pèse **16 %** des
positions.

L'expérience 1 enchâsse les mêmes phrases dans le prompt 2AFC. Les tokens perturbés y
sont aux positions ~40 à ~80, et **aucun n'est jamais en position 0**.

`s(ℓ,v)` est donc estimée sur une distribution de tokens **qui n'apparaît pas dans
l'expérience**. Comme `z = α / s(ℓ,v)`, tous les résultats en `z` du balayage
`full32_all` héritent du défaut.

Ce n'est ni un défaut d'implémentation ni un problème de données. C'est le contexte de
présentation provisoire que la configuration signale elle-même, et que le validateur du
protocole refuse déjà au moment du gel. **Le correctif ne demande aucune modification du
moteur de calibration** : le mode `external_manifest`, prévu exactement pour ça, existe
déjà.

En cherchant la cause, une seconde anomalie est apparue, plus lourde de conséquences et
**indépendante du correctif** : les vecteurs conceptuels sont quasi colinéaires à la
direction du token de puits (cosinus 0,70 à 0,99, contre 0,016 attendu pour une direction
aléatoire). Dans les blocs précoces, injecter un « concept » revient donc surtout à
injecter la direction d'activation massive. Voir §4.

---

## 1. Ce que le protocole veut dire par `s(ℓ,v)`

Le cadrage (§1) définit `s(ℓ,v)` comme

> l'échelle de variabilité des activations **sans intervention** projetées dans cette
> direction

et `z = α / s(ℓ,v)` comme l'amplitude rapportée à cette échelle naturelle. L'intention
est claire : `z = 1` doit vouloir dire « une perturbation de l'ordre de ce que ces
activations font naturellement ».

Pour que ça ait un sens, l'échantillon sur lequel `s` est estimée doit être la
distribution des activations **que l'expérience perturbe**. C'est cette condition qui
est violée.

---

## 2. Le problème

### 2.1 Ce que calibre l'expérience 0

`configs/experiment_0_calibration/development_full.yaml` :

```yaml
presentation:
  context_id: isolated_sentence_development
  mode: template_per_sentence
  template: "{sentence}"
  position_policy: all_sentence_tokens
```

Chaque phrase est rendue seule. `step_01_02_collect_natural_activations.py` tokenise
avec `add_special_tokens=False` — donc pas de BOS — puis collecte **tous** les tokens de
la phrase. Chaque phrase étant sa propre séquence, son premier token est à la
**position 0**.

En Llama, la position 0 d'une séquence est le puits d'attention : n'importe quel token
qui s'y trouve accumule une activation de norme énorme, parce que l'attention doit bien
déverser sa masse de probabilité quelque part. C'est le phénomène d'*activation
massive*.

L'échantillon compte 616 positions pour 100 phrases, soit **6,16 tokens par phrase**.
Un token de position 0 par phrase fait donc **16,2 %** de l'échantillon.

### 2.2 Ce que perturbe l'expérience 1

`build_localization_prompt()` (`code/experiments/experiment1_psychometrics.py`) construit
le prompt 2AFC complet : gabarit de conversation, consigne de plusieurs lignes, les deux
phrases préfixées par `A) ` et `B) `, puis `The answer is`. Le `token_range` renvoyé
**exclut l'étiquette** et ne couvre que les tokens de la phrase.

Ces tokens sont aux positions ~40 à ~80 du prompt. **Aucun token perturbé n'est jamais
en position 0.**

### 2.3 La conséquence arithmétique

`s_SD` est écrasée par les 16 % de tokens de puits, dont la projection dépasse les
autres de deux à trois ordres de grandeur.

Cet effet ne frappe **pas** les familles de manière uniforme, et la raison n'est pas une
différence de norme des vecteurs : toutes les directions sont normalisées à 1. Elle est
géométrique, et elle est mesurée en §4. Le rapport d'échelle entre familles —
précisément ce que la division par `s` est censée éliminer — est donc lui-même un
artefact.

---

## 3. La preuve

Mesure directe (`code/analysis/probe_calibration_context.py`, jobs 8174 et 8175,
résultats dans `results/experiment1/context_probe/`). Les **mêmes** 100 phrases et les
**mêmes** directions sont projetées dans les deux contextes.

### 3.1 Le token extrême est en position 0, et nulle part ailleurs

Pour `concept__block_01__recursion`, contexte isolé :

| | valeur |
|---|---|
| \|projection\| maximale **en position 0** | **582,3** |
| \|projection\| maximale **partout ailleurs** | **0,399** |
| part de tokens en position 0, contexte **isolé** | **0,162** |
| part de tokens en position 0, contexte **prompt** | **0,000** |

Un facteur **1460** entre la position 0 et tout le reste. Et la part de 0,162 reproduit
la masse de queue de 0,159 estimée indépendamment sur `directional_scales.csv`, par
`f = |moyenne − médiane| / |extrême − médiane|` — laquelle vaut 0,159 dans les **trois**
familles, concept comme aléatoire, ce qui excluait déjà une propriété du contenu
conceptuel.

### 3.2 La queue lourde disparaît dans le contexte comportemental

Médiane sur les directions de chaque famille :

| bloc | famille | SD isolé | SD/MAD isolé | SD prompt | SD/MAD prompt |
|---:|---|---:|---:|---:|---:|
| 1 | concept | 198,67 | 2451 | **0,0248** | **0,93** |
| 1 | aléatoire | 3,55 | 141 | 0,0203 | 1,00 |
| 5 | concept | 184,21 | 1196 | 0,1115 | 1,11 |
| 5 | aléatoire | 1,95 | 25,4 | 0,0562 | 1,06 |
| 9 | concept | 160,82 | 623 | 0,1876 | 0,98 |
| 9 | aléatoire | 1,70 | 15,5 | 0,0824 | 0,96 |
| 13 | concept | 146,98 | 429 | **0,2511** | **0,91** |
| 13 | aléatoire | 1,62 | 12,6 | 0,1069 | 0,97 |

Dans le contexte que l'expérience 1 perturbe réellement, `SD/MAD` vaut **0,91 à 1,11**
pour toutes les familles : la distribution des projections y est essentiellement
gaussienne. SD chute d'un facteur ~800 à ~8000 par rapport au contexte isolé.

### 3.3 Le rapport d'échelle entre familles s'effondre

C'est la quantité qui distordait la comparaison :

| bloc | isolé, par SD | isolé, par MAD | prompt, par SD | prompt, par MAD |
|---:|---:|---:|---:|---:|
| 1 | 55,9 | 3,21 | **1,22** | 1,28 |
| 5 | 94,7 | 1,88 | **1,98** | 1,66 |
| 9 | 94,4 | 2,21 | **2,28** | 2,33 |
| 13 | 90,6 | 2,82 | **2,35** | 2,02 |

Rapport concept/aléatoire. Dans le bon contexte il vaut **1,2 à 2,4**, et les deux
estimateurs s'accordent. Le facteur 56–95 mesuré sur la calibration est **entièrement**
l'effet du token de position 0.

---

## 4. La cause profonde : les vecteurs conceptuels sont presque colinéaires au token de puits

Le rapport de 56 à 95 entre l'échelle conceptuelle et l'échelle aléatoire demandait une
explication. Toutes les directions étant de norme 1, il ne peut venir que de
l'**orientation** des vecteurs. Mesure (jobs 8177 et 8178,
`code/analysis/probe_sink_alignment.py`) :

### 4.1 Le token de position 0 porte une norme d'activation démesurée

| bloc | ‖h‖ en position 0 | ‖h‖ ailleurs | rapport |
|---:|---:|---:|---:|
| 1 | 546,8 | 1,70 | **321** |
| 5 | 546,8 | 5,03 | **109** |
| 9 | 547,0 | 7,69 | **71** |
| 13 | 550,4 | 9,65 | **57** |

Norme quasi constante autour de 547 à toutes les profondeurs, contre 1,7 à 9,7 pour les
autres tokens. C'est la signature de l'activation massive.

### 4.2 Les directions conceptuelles sont alignées sur cette direction, les aléatoires non

Cosinus absolu entre chaque direction et l'activation moyenne de position 0. Pour une
direction aléatoire en dimension 4096, l'attendu est `1/√4096 = 0,0156`.

| bloc | Dust | Satellites | recursion | shutdown | aléatoires (3) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,982 | 0,632 | **0,994** | 0,983 | 0,011 – 0,027 |
| 5 | 0,928 | 0,340 | 0,958 | 0,894 | 0,005 – 0,011 |
| 9 | 0,811 | 0,255 | 0,896 | 0,778 | 0,001 – 0,035 |
| 13 | 0,744 | 0,205 | 0,856 | 0,699 | 0,007 – 0,011 |

Les directions aléatoires tombent exactement sur l'attendu isotrope : elles ne « voient »
pas le token de puits. Les directions conceptuelles en sont **quasi colinéaires**. Le
rapport d'échelle s'en déduit directement : `0,86 / 0,0156 ≈ 55`, ce qui reproduit les
56–95 mesurés.

### 4.3 Pourquoi, et ce que cela implique au-delà de la calibration

Les vecteurs conceptuels sont construits comme des **moyennes d'états cachés**
(`concept_vector_type: avg`). Or la composante d'activation massive est présente dans
tout état caché moyenné, et elle domine par sa norme. Chaque vecteur « concept » hérite
donc d'une composante de puits d'attention écrasante.

Un vecteur à `|cos| = 0,99` de la direction de puits n'a que `√(1 − 0,99²) ≈ 14 %` de sa
norme dans un sous-espace spécifique au concept. **Dans les blocs précoces, injecter un
« concept » revient surtout à injecter la direction d'activation massive.**

Cela déborde du présent correctif, et l'éclaire en retour. Trois recoupements :

- Cela explique l'axe unique observé dans l'analyse de l'expérience 1 : PC1 capte 61 à
  83 % de la variance entre les quatre vecteurs conceptuels, parce que **PC1 est la
  direction d'activation massive**, non un axe sémantique.
- Cela explique l'anomalie Dust, seul concept inerte et seul à charger négativement sur
  PC1 : il est aligné sur la direction de puits avec le signe opposé aux autres.
- Cela explique que Satellites soit à part : c'est le concept le **moins** aligné
  (0,21–0,63), et celui dont le chargement sur PC1 est quasi nul.

**Ce point n'est pas réglé par la recalibration.** Recalibrer corrige `s(ℓ,v)` ; cela ne
change pas ce que *sont* les directions conceptuelles. Les deux questions sont
distinctes et doivent être traitées séparément (§9).

---

## 5. Pourquoi ce n'est ni l'implémentation ni les données

**Ce n'est pas un défaut d'implémentation.** Vérifications faites, toutes négatives :
l'écart-type est calculé correctement ; `direction_bank.scale()` applique bien l'échelle
d'une direction à la famille qui la porte ; `add_special_tokens=False` empêche l'ajout
d'un BOS parasite ; `step_01_02` vérifie même l'absence de dérive de tokenisation avant
de collecter. Le code fait exactement ce qu'on lui demande.

**Ce n'est pas un problème de données.** Les 100 phrases de `LOCALIZATION_SENTENCES`
sont les mêmes des deux côtés.

**C'est le contexte de présentation, et il est annoncé comme provisoire.** En-tête de
`development_full.yaml` :

```yaml
# Experiment 0 — natural directional-scale calibration.
# This is deliberately marked development: the isolated-sentence context must
# be replaced by the exact behavioural prompt context before protocol freezing.
```

Le fichier porte `status: development`, et `protocol_config.py` **refuse de geler** un
protocole qui ne serait pas en contexte comportemental :

```python
if self.protocol_status == "frozen" and (
    ...
    or self.context_id.endswith("development")
    or self.presentation_mode != "external_manifest"
    ...
):
    raise ValueError(...)
```

Les garde-fous ont fonctionné. Le défaut ne pouvait survivre au gel ; il n'existe que
parce que le balayage a été lancé sur une calibration en statut `development`, ce qui a
exigé `--allow_calibration_mismatch`.

---

## 6. Un piège de cadrage à ne pas confondre avec le problème

> « Le balayage en z n'a jamais mis les familles sur la même échelle physique. »

**Ce n'est pas le défaut, et ce n'était pas le but.** Mettre les familles à la même
amplitude physique est le rôle de l'appariement **en α**. Le cadrage définit deux
paramétrisations précisément pour qu'elles diffèrent, et H1c énonce que l'appariement en
`z` **modifie** les écarts observés à `α` apparié. Si `z` livrait la même amplitude
brute à toutes les familles, il serait identique à `α` et H1c serait vide.

Qu'une direction conceptuelle reçoive plus d'amplitude brute à `z` égal est donc le
mécanisme attendu. Le défaut est que `s` a été estimée sur la mauvaise distribution — et
accessoirement que l'écart ainsi produit (56–95×) est assez grand pour qu'aucune famille
témoin ne soit testée près de sa transition.

---

## 7. Ce qui est invalidé, ce qui ne l'est pas

| | statut |
|---|---|
| Résultats en **`z`** du balayage `full32_all` | **Non interprétables**, ni pour confirmer ni pour infirmer H1c. |
| Résultats en **`α`** | **Intacts.** Ils n'appellent aucune échelle calibrée. |
| Seuils `z₇₅` | Sans objet — ils étaient déjà écartés pour non-monotonie des courbes. |
| Ordonnancement des familles | **Robuste.** `concept > bruit ≈ dropout > aléatoire` tient à α apparié, à z_MAD et à amplitude réalisée appariée. |
| Amplitude de l'avantage conceptuel | **Non établie.** Elle varie d'un facteur 2,2 selon la paramétrisation. |
| Coupure de profondeur au bloc 14 | **Intacte.** Mesurée à α apparié. |

---

## 8. Le correctif : comment recalculer `s(ℓ,v)`

### 8.1 Le moteur de calibration accepte déjà tout

Aucune modification n'est nécessaire dans `code/experiment_0_calibration/`.
`prepare_material.py` expose le mode qu'il faut, et sa docstring dit pourquoi il existe :

> Development mode renders one template per sentence. A frozen run must use an external
> JSONL manifest with rows shaped as
> `{context_id, rendered_text, targets: [{sentence_id, char_start, char_end}]}`.
> **This supports paired prompts and repeated sentence presentations** without coupling
> Experiment 0 to behavioural-task code.

Vérifications faites sur le chemin d'ingestion, toutes concluantes :

| Ce qu'il faut | Ce que le code fait déjà |
| --- | --- |
| Deux phrases par prompt | `targets` est une liste, parcourue par `target_index` |
| Retrouver l'emplacement plus tard | chaque observation enregistre son `target_index` |
| Ne pas se tromper de tokens | `rendered[char_start:char_end] != sentence` lève une erreur |
| Refuser un token à cheval | `overlaps and not contained` lève une erreur |
| Une phrase dans plusieurs prompts | `observation_id` est préfixé par `context_id`, donc pas de collision |
| Bootstrap correct | `build_phrase_bootstrap_plan` groupe par `sentence_id` : toutes les occurrences d'une phrase se déplacent ensemble, ce qui est le bon regroupement |

### 8.2 Le seul travail : produire le manifeste

Une ligne JSON par prompt :

```json
{"context_id": "loc2afc__pair_00__order_0__AB",
 "rendered_text": "<le prompt 2AFC complet, rendu par build_localization_prompt>",
 "targets": [{"sentence_id": "localization_000", "char_start": 312, "char_end": 333},
             {"sentence_id": "localization_001", "char_start": 337, "char_end": 365}]}
```

Les bornes en caractères existent déjà dans `build_localization_prompt()`
(`code/experiments/experiment1_psychometrics.py`), qui calcule `start_char` et
`end_char` **hors étiquette** `A) ` / `B) ` avant de les convertir en tokens. Il suffit
de les émettre au lieu de les jeter.

Puis pointer la configuration dessus :

```yaml
presentation:
  context_id: localization_2afc_prompt
  mode: external_manifest
  manifest: data/experiment_0_calibration/contexts_2afc.jsonl
  position_policy: all_sentence_tokens
```

`all_sentence_tokens` reste correct : la politique sélectionne les tokens des `targets`,
et les `targets` ne couvrent plus que les phrases dans leur prompt. Aucun token de
position 0 n'y entre.

### 8.3 Quatre décisions de conception

**Quelles phrases ?** Les 100 du corpus, comme aujourd'hui — et non les seules 10 des
cinq paires que l'expérience 1 utilise. L'échelle est une propriété de (bloc, direction),
pas d'une paire de phrases ; se restreindre aux paires du plan comportemental ferait
tomber N de 616 à ~62 et coupleraient la calibration à la tâche, ce que la docstring
citée plus haut cherche précisément à éviter.

**Combien de prompts ?** Apparier les 100 phrases en 50 paires, et rendre chaque paire
dans les 2 ordres × 2 ordres d'étiquettes. Chaque phrase apparaît alors 4 fois, dont deux
fois à chaque emplacement. Cela donne **200 passes avant et N = 2464 positions**, soit
quatre fois l'échantillon actuel pour un coût négligeable, et un plan équilibré en
emplacement et en étiquette.

**Une échelle par emplacement ?** Non. Dans le prompt les deux phrases occupent des
positions différentes ; la question est de savoir si l'échelle en dépend. Mesuré
(job 8176, `slot_probe.json`) :

| bloc | famille | SD emplacement 0 | SD emplacement 1 | rapport | SD `AB` | SD `BA` | rapport |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | concept | 0,0231 | 0,0265 | 1,15 | 0,0248 | 0,0251 | 1,01 |
| 5 | concept | 0,1129 | 0,1090 | 0,96 | 0,1115 | 0,1140 | 1,02 |
| 9 | concept | 0,2089 | 0,1683 | 0,81 | 0,1876 | 0,1934 | 1,03 |
| 13 | concept | 0,2774 | 0,2296 | 0,83 | 0,2511 | 0,2624 | 1,04 |
| 13 | aléatoire | 0,1012 | 0,1033 | 1,02 | 0,1069 | 0,0980 | 0,92 |

La dépendance à l'emplacement plafonne à ~20 %, et l'ordre des étiquettes ne fait rien
(0,92–1,04). Une échelle unique par `(bloc, direction)`, poolée sur les deux emplacements
et les deux ordres, suffit : 20 % est petit devant le pas ×2 de la grille de doses. La
variation résiduelle se rapporte en sensibilité — `target_index` est déjà enregistré par
observation — et ne complique pas le protocole.

**SD ou MAD ?** SD, comme prévu au protocole. Dans le contexte comportemental les deux
estimateurs coïncident (rapport 0,91–1,11, §3.2) et SD est six fois plus précise
(CV bootstrap 0,008 contre 0,049). MAD redevient le contrôle de robustesse peu coûteux
que prévoit la section 5.10 du cadrage.

### 8.4 Le coût

L'expérience 0 ne fait que des passes avant sans intervention. La sonde a mesuré les deux
contextes sur 100 phrases et 4 blocs en **moins de dix minutes sur un seul GPU** ; les 32
blocs et les 20 directions de la calibration complète restent du même ordre, et les 200
prompts du §8.3 doublent seulement le nombre de passes. Le coût n'est pas un argument
pour conserver le contexte isolé.

### 8.5 La grille `z` devient enfin faisable — mais doit être refaite

C'est le bénéfice le moins évident de la recalibration. Le rapport d'échelle
concept/aléatoire tombe de **54×** à **1,2–2,4×** : une grille `z` **partagée par les
quatre familles** redevient donc possible, ce qu'elle n'était pas. En revanche l'étendue
de la grille actuelle est à jeter.

| bloc | s concept | s aléat. | ratio | `z` pour couvrir α ∈ [0,25 ; 128] | octaves |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,0248 | 0,0203 | 1,22 | 10 → 6310 | 9,3 |
| 5 | 0,1115 | 0,0562 | 1,98 | 2,2 → 2277 | 10,0 |
| 9 | 0,1876 | 0,0824 | 2,28 | 1,3 → 1553 | 10,2 |
| 13 | 0,2511 | 0,1069 | 2,35 | 1,0 → 1198 | 10,2 |

Soit environ **10 octaves par bloc**, ~13 points au pas ×2. La grille reste dépendante du
bloc, puisque l'échelle croît avec la profondeur, mais plus de la famille. Pour mémoire,
la grille actuelle `z ∈ [0,01 ; 20,48]` plafonnerait à α ≈ 0,5 au bloc 1 et α ≈ 5 au
bloc 13 : elle ne mesurerait rien.

**Règle de vérification, à appliquer avant de publier quoi que ce soit en `z`.** Contrôler
que la dose maximale délivre, pour **chaque** famille, une amplitude supérieure à son
propre seuil mesuré sur la grille α. C'est exactement ce contrôle qui manquait et qui a
produit l'artefact d'étendue sur Qwen — où la grille dépassait le seuil du concept de 6 à
14 fois et ratait celui de chaque témoin de 5 à 20 fois.

### 8.6 Vérifier que la recalibration a bien pris

Rejouer `code/analysis/probe_calibration_context.py` sur l'échantillon effectivement
recalibré. Deux critères :

- `s_SD / s_MAD ≈ 1` pour toutes les familles ;
- masse de queue × tokens/phrase nettement sous 1.

Si ce n'est pas le cas, le manifeste ne cible pas les bons tokens — le plus probable
étant des bornes en caractères décalées, que la vérification
`rendered[char_start:char_end] != sentence` de `prepare_material.py` aurait normalement
déjà attrapée.

## 9. Ce que devient l'analyse MAD

La §5.10 du cadrage prévoit MAD comme analyse de sensibilité. Le diagnostic ci-dessus en
change le statut, dans les deux sens.

**MAD n'est pas le correctif.** Dans le contexte comportemental, `SD ≈ MAD` (rapport
0,91–1,11). Le défaut n'était pas l'estimateur mais l'échantillon sur lequel on
l'applique. Une fois la recalibration faite, SD — six fois plus précis, CV bootstrap
0,008 contre 0,049 — reste l'estimateur primaire que le cadrage lui assigne, et MAD
redevient le contrôle de robustesse peu coûteux prévu.

**Mais MAD a une valeur diagnostique réelle, et une propriété remarquable.** `s_MAD`
mesurée dans le contexte isolé est proche de `s_SD` mesurée dans le bon contexte :

| bloc | `s_MAD` isolé | `s_SD` prompt |
|---:|---:|---:|
| 9 (concept) | 0,262 | 0,188 |
| 13 (concept) | 0,362 | 0,251 |

**MAD approximait accidentellement la bonne quantité**, en jetant le token qui n'aurait
pas dû être dans l'échantillon. C'est pourquoi la réanalyse MAD rapproche les résultats
de la comparaison à amplitude physique appariée : elle corrigeait le bon défaut, pour
une raison indirecte. C'est aussi ce qui a permis de détecter le problème.

---

## 10. Plan d'action

1. **Générer le manifeste 2AFC** — 50 paires × 2 ordres × 2 ordres d'étiquettes, bornes
   en caractères hors étiquette — et basculer la calibration en `external_manifest`
   (§8.2). C'est le correctif principal, et une condition du gel. Aucune modification du
   moteur de calibration (§8.1).
2. **Ne pas interpréter les résultats en `z`** de `full32_all` d'ici là, dans aucun sens
   (§7).
3. **Refaire la grille `z`** avant toute nouvelle exécution : ~10 octaves par bloc, de
   `z ≈ 1` à `z ≈ 6300` selon la profondeur (§8.5). Ce défaut est **indépendant** du
   précédent et survit à la recalibration ; en revanche, celle-ci rend enfin possible une
   grille **partagée par les quatre familles**, le rapport d'échelle tombant de 54× à
   1,2–2,4×.
4. **Produire la réanalyse MAD** comme diagnostic immédiat, en reliant les essais par
   leur `realized_amplitude` plutôt que par la dose demandée. Coût GPU nul.
5. **Refaire tourner la sonde** après recalibration, avec les deux critères du §8.6 :
   `jobs/probe_calibration_context.sbatch`.

### Chantier distinct, à ouvrir séparément

6. **Traiter la colinéarité des vecteurs conceptuels avec la direction de puits** (§4).
   Ce n'est pas un problème de calibration et la recalibration ne le corrige pas. Piste
   la plus directe : retirer la composante de puits des vecteurs `avg` avant de les
   utiliser comme directions d'injection, puis vérifier que PC1 cesse de capter 61–83 %
   de la variance entre concepts. Tant que ce n'est pas fait, l'écart concept/témoins de
   l'expérience 1 ne peut pas être attribué au contenu conceptuel.
7. **Rejouer le contrôle `−Dust`** après ce traitement. Il ne tranchait
   « axe signé contre concept » que si l'axe en question reste défini ; si l'axe est la
   direction de puits, le contrôle change de sens.

---

## Références

| | |
|---|---|
| Analyse complète de l'expérience 1 | `docs/livrables/experiment1-llama-analyse.tex`, §« D'où vient le problème » et §« Mesure directe des deux contextes » |
| Script de sonde | `code/analysis/probe_calibration_context.py`, `jobs/probe_calibration_context.sbatch` |
| Résultats | `results/experiment1/context_probe/` (jobs 8174, 8175, 8176, 8177) |
| Sonde d'alignement au puits | `code/analysis/probe_sink_alignment.py`, `jobs/probe_sink_alignment.sbatch` |
| Configuration en cause | `configs/experiment_0_calibration/development_full.yaml` |
| Validateur du gel | `code/experiment_0_calibration/protocol_config.py` |
