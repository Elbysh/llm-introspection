# Expérience 3 — variabilité entre concepts, directions et couches

Doc de cadrage section 7. Les essais sont ceux de l'expérience 1 : le balayage fait
déjà varier `direction_id` à l'intérieur de chaque famille, donc une lecture par
direction des mêmes lignes *est* l'expérience 3, et aucune passe GPU nouvelle n'a été
nécessaire. Ce que `summary.json` ne pouvait pas dire, c'est précisément la question :
il regroupe tous les concepts en une seule courbe `concept` par (couche, appariement).

Analyse : `code/analysis/experiment3_variability.py`.
Résultats : `results/experiment3/`.

| bras | modèle | appariement | concepts | couches |
|---|---|---|---|---|
| `llama_z` | Llama-3.1-8B | z | **10** | 0–31 |
| `llama_alpha` | Llama-3.1-8B | alpha | **10** | 0–30 |
| `llama_alpha_10concepts` | Llama-3.1-8B | alpha | **10** | 3, 16, 28 |
| `qwen_z` | Qwen3.8-27B | z | 4 | 3, 6, 8, 10, 12, 16 |
| `qwen_alpha` | Qwen3.8-27B | alpha | 4 | 3, 6, 8, 10, 12, 16, 32, 56 |

## 1. La mesure

La section 7.5 demande une précision, un contraste et un seuil individuel à 75 %. Le
seuil est presque toujours indisponible ici : la précision 2AFC brute est bloquée près
du hasard par le biais de réponse du modèle, et les courbes *groupées* ne traversent
75 % que dans 0 à 5 couches sur 32. Par direction, une cellule a le quart des essais.
Le seuil est donc rapporté là où la courbe encadre réellement 75 % — 26/352, 11/348,
16/66, 34/88 et 1/51 courbes selon le bras — mais la mesure de travail est le contraste
de localisation apparié de la section 5.8,

    S = (contraste quand A est ciblée − contraste quand B est ciblée) / 2

sur les deux essais qui partagent tout sauf la phrase touchée. Tout décalage
indépendant de la cible s'annule, ce qui rend S lisible là où la précision ne l'est
pas. C'est l'estimateur et la clé d'appariement de `experiment1_localization_report.py`.

Les trois doses de la section 7.3 sont fixées avant la lecture par direction, à partir
de la courbe groupée : le seuil global quand il est rapportable (bras alpha de Qwen,
32,2), sinon la dose testée où |S| groupé est maximal. Ce sont toujours des valeurs de
la grille, jamais interpolées.

## 2. La variance entre directions se lit par couche, pas en moyenne sur la profondeur

C'est le point méthodologique qui commande tout le reste, et il est facile à manquer.

La profondeur est très majoritairement inerte. Sur Llama, au-delà du bloc ~14, toutes
les familles retombent à S ≈ 0 ; sur le plan à 10 concepts, seul le bloc 3 est vivant
(S moyen 0,629) tandis que les blocs 16 et 28 donnent −0,004 et 0,009. Dans une couche
morte, toutes les directions valent 0 et l'écart entre elles s'annule avec le reste.

Un écart-type entre directions pris sur l'ensemble des couches est donc divisé par le
nombre de couches mortes que *ce balayage-là* contient. Ce n'est pas une propriété des
concepts, c'est une propriété du plan de balayage. Comparer deux familles sur cette
base est trompeur dès que leurs profils en profondeur diffèrent :

| bras | ratio écart-type concept / aléatoire, **groupé** | le même, **à la couche de pic de chaque famille** |
|---|---|---|
| `llama_z` | 2,6× | 0,6× |
| `llama_alpha` | 3,1× | 1,1× |
| `qwen_z` | 9,2× | 3,6× |
| `qwen_alpha` | 3,7× | 0,3× |
| `llama_alpha_10concepts` | 1,5× | 1,8× |

Lu en groupé, on conclurait que les concepts se dispersent 4 à 11 fois plus que des
directions aléatoires. Lu à la couche où chaque famille agit réellement, le rapport
tombe entre 0,3× et 3,6× : selon le bras, les directions aléatoires se dispersent
autant ou davantage. **La conclusion « les concepts sont bien plus hétérogènes que des
directions aléatoires » n'est pas soutenue ; elle était un artefact de regroupement.**
`variability_report.json` porte désormais `by_layer` pour chaque famille, avec le S
moyen et l'écart-type entre directions couche par couche.

## 3. Composantes de variance

Décomposition en moyennes sur les cellules (direction × couche × phrase), qui est la
partie fixe du modèle mixte de la section 7.6 : le plan est équilibré par construction,
donc les termes sont orthogonaux et les parts somment à 1 sans ajustement.

| bras | famille | direction | couche | phrase | interaction |
|---|---|---|---|---|---|
| Llama z | concept | 0,046 | 0,685 | 0,018 | 0,251 |
| Llama z | random | 0,010 | 0,622 | 0,018 | 0,350 |
| Llama alpha | concept | 0,051 | 0,704 | 0,020 | 0,225 |
| Llama alpha | random | 0,008 | 0,624 | 0,017 | 0,350 |
| Llama alpha, 10 concepts | concept | 0,055 | 0,732 | 0,023 | 0,191 |
| Llama alpha, 10 concepts | random | 0,030 | 0,851 | 0,018 | 0,101 |
| Qwen z | concept | 0,288 | 0,357 | 0,092 | 0,263 |
| Qwen z | random | 0,062 | 0,295 | 0,071 | 0,572 |
| Qwen alpha | concept | 0,078 | 0,810 | 0,020 | 0,092 |
| Qwen alpha | random | 0,013 | 0,697 | 0,013 | 0,276 |

**La couche domine partout** (0,30 à 0,85). C'est la structure réelle : une fenêtre
superficielle où la perturbation agit, puis une profondeur inerte. La part « direction »
de ce tableau subit la dilution de la section 2 et doit se lire comme une borne basse.

**La phrase ne porte presque rien** (0,011 à 0,092) : les 5 paires du corpus ne sont pas
une source de variance sur cette tâche.

**L'interaction direction × couche est substantielle** (0,092 à 0,572) et c'est
exactement la mise en garde de la section 7.7 : une différence entre familles ne peut
pas être attribuée à l'architecture sans nommer le site d'injection. C'est aussi la
signature de la structure vivant/mort — les directions diffèrent là où ça mord et pas
ailleurs.

## 4. Ce que le panel complet a changé — y compris dans le mauvais sens

Les jobs 8425 et 8426 ont balayé les six concepts et les sept directions aléatoires
manquants sur les 32 blocs et les deux appariements. Les panels Llama sont désormais
**10 concepts × 10 directions aléatoires**. Le plan avant/après, sur S moyen groupé sur
tous les blocs et toutes les doses :

| appariement | panel | concept | aléatoire | écart |
|---|---|---|---|---|
| z | 4 × 3 | +0,1185 | +0,0727 | **0,0458** |
| z | 10 × 10 | +0,1175 | +0,0797 | **0,0378** |
| alpha | 4 × 3 | +0,1373 | +0,1008 | **0,0365** |
| alpha | 10 × 10 | +0,1401 | +0,1104 | **0,0297** |

**L'écart concept/contrôle se resserre de 17 % en z et de 19 % en alpha.** Une note
antérieure de ce dépôt prévoyait l'inverse — un écart plus large d'environ 70 % — à
partir d'une extrapolation du bloc 3 du balayage `main`. Cette prévision était fausse, et
sur la direction, pas seulement sur l'amplitude. Deux raisons :

- Elle tenait la ligne de base aléatoire pour fixe, faute de données sur 0003–0009. Or
  les trois directions d'origine sont parmi les plus faibles des dix : moyenne +0,0727
  contre +0,0827 pour les sept ajoutées, soit une base sous-estimée de 10 %.
- Elle lisait `Dust` au bloc 3, où il est à son plus extrême (0,011 contre 0,63–0,87).
  Groupé sur toute la profondeur, `Dust` vaut +0,041 contre +0,11–0,15 pour les autres :
  bien moins atypique, et les quatre concepts d'origine tombent alors presque exactement
  sur la moyenne de population (+0,1185 contre +0,1175).

Le panel à quatre concepts et trois directions aléatoires **flattait** donc la séparation
concept/contrôle d'environ 20 %, au lieu de la minorer. La conclusion qualitative de
l'expérience 1 tient — concept reste au-dessus des trois familles témoins sous les deux
appariements — mais son amplitude était surestimée.

## 5. Une dissociation minoritaire, pas une hétérogénéité générale

Sur le panel complet — 10 concepts, 32 blocs, appariement z, S moyen groupé sur tous les
blocs et toutes les doses :

| concept | S | | direction aléatoire | S |
|---|---|---|---|---|
| shutdown | 0,1538 | | 0003 | 0,1006 |
| Origami | 0,1475 | | 0009 | 0,0914 |
| Satellites | 0,1417 | | 0005 | 0,0904 |
| appreciation | 0,1410 | | 0007 | 0,0797 |
| recursion | 0,1372 | | 0002 | 0,0777 |
| betrayal | 0,1270 | | 0008 | 0,0732 |
| fibonacci_numbers | 0,1155 | | 0004 | 0,0732 |
| Illusions | 0,1122 | | 0000 | 0,0705 |
| **Trumpets** | **0,0582** | | 0006 | 0,0701 |
| **Dust** | **0,0413** | | 0001 | 0,0697 |

Huit concepts sur dix tiennent dans 0,1122–0,1538, écart-type **0,0149**, soit un noyau
plus homogène que les dix directions aléatoires entre elles (0,0697–0,1006, écart-type
0,0108, du même ordre). Deux décrochent : `Trumpets` à 0,0582 et `Dust` à 0,0413, tous
deux **sous la plus faible des dix directions aléatoires**. Un concept qui décroche ne
fait donc pas seulement moins bien que les autres concepts : il fait moins bien qu'une
direction tirée au hasard.

**Cela répond à la question laissée ouverte par le panel à 3 couches.** La dissociation de
`Dust` n'est pas un accident du bloc 3 : elle tient sur toute la profondeur, avec la même
structure — un corps homogène et une minorité qui ne répond pas. C'est le constat de
Macar et al. que cite la section 7.2, précisé : ce n'est pas « les concepts varient
beaucoup », c'est « la plupart se comportent pareil, et certains ne répondent pas du
tout ». Le panel à 4 concepts ne pouvait pas le montrer, `Dust` y pesant 1 cas sur 4.

Le balayage `main` le montrait déjà au bloc 3, sur sa seule couche vivante (S moyen
0,629 ; blocs 16 et 28 à −0,004 et 0,009) : huit concepts dans 0,631–0,868, `Trumpets` à
0,281 et `Dust` à 0,011. Les deux lectures concordent.

## 6. Quel concept décroche dépend du modèle

À la couche de pic de chaque bras, sur les 4 concepts communs :

| bras | ordre, du plus faible au plus fort |
|---|---|
| Llama z (bloc 8) | **Dust 0,535** < recursion 0,800 < shutdown 0,846 < Satellites 0,974 |
| Llama alpha (bloc 10) | **Dust −0,041** < Satellites 0,747 < shutdown 0,990 < recursion 0,997 |
| Qwen z (bloc 10) | recursion 0,086 < **Dust 0,122** < Satellites 0,195 < shutdown 0,275 |
| Qwen alpha (bloc 3) | shutdown 0,964 < Satellites 1,026 < **Dust 1,046** < recursion 1,223 |

`Dust` est le concept le plus faible sur Llama sous les deux appariements, et il est
au milieu du classement sur Qwen — troisième sur quatre en alpha, au-dessus de
`shutdown` et de `Satellites`. Le décrochage de `Dust` est donc propre à Llama.

La variabilité entre concepts est réelle et reproductible sous les deux variables de
dose à l'intérieur d'un modèle, mais *quel* concept décroche est une propriété du
modèle, pas du concept. Cela renforce le constat de la section 7.2 : non seulement la
norme du vecteur ne le prédit pas, mais l'identité du concept ne se transporte pas d'un
modèle à l'autre.

## 7. Pourquoi le bras alpha est ici

La section 7.3 ne demande que des doses en z. Le bras alpha est analysé en plus parce
qu'il est le contrôle qui rend le reste lisible : à
`experiment1_psychometrics.py:852`, sous `matching == "alpha"` l'amplitude *est* la
valeur de grille et `calibration.scale()` n'est jamais consultée. L'appariement alpha ne
dépend donc d'aucune estimation de s(ℓ,v), et la structure de variance qu'il donne ne
peut être ni un artefact de calibration ni un artefact de la plage de la grille z. Elle
reproduit celle du bras z sur Llama : part de direction 0,102 contre 0,088, écart-type
groupé entre concepts 0,135 contre 0,137, et le même concept en dernier.

C'est aussi ce qui rend `main` exploitable, voir la section suivante.

## 8. Provenance des bras alpha

Les lignes alpha proviennent de balayages antérieurs à la recalibration 2AFC
(`full32_all`, `qwen38_all_layers`) et, pour le plan à 10 concepts, d'un balayage
antérieur à la refonte de la calibration elle-même (`main`, qui référence
`configs/calibration/full.yaml` et nomme ses directions `concept__l3__Dust` au lieu de
`concept__block_03__Dust`). Aucun de ces défauts n'atteint le bras alpha, qui n'invoque
aucune échelle calibrée.

Vérifié plutôt que supposé : sur les 4 concepts communs, aux 3 couches communes et à la
même grille de doses, `main` et `full32_all` donnent les *mêmes* valeurs à trois
décimales — Satellites 0,297 / 0,297, shutdown 0,239 / 0,239, recursion 0,235 / 0,235,
Dust 0,021 / 0,021, r = 1,00. Les deux générations de pipeline mesurent la même chose
sur ce bras ; la différence de nommage et de config de calibration est cosmétique pour
alpha.

Seules les lignes alpha ont été extraites et versionnées, comme répertoires de run à
part entière (`results/experiment3/*_source/`) ; le bras z défectueux de ces balayages
reste délibérément hors du dépôt.

Le bras alpha de Llama est lu sur les blocs **0–30**, pas 0–31. `full32_all` n'a jamais
balayé le bloc 31 pour la famille concept : le vecteur conceptuel du bloc 31 demande
`data/saved_vectors/llama/*_32_*.pt`, alors non versionné, et la norme régénérée ne
reproduisait pas celle qu'avait enregistrée l'ancienne calibration. La famille aléatoire,
elle, couvre bien les 32 blocs. Décomposer les deux familles sur des ensembles de couches
différents rendrait leurs parts de variance non comparables, donc `--exclude_layers 31`
les ramène au même support. Le bloc 31 ne perd rien d'interprétable : aucune couche
d'attention ne le suit, donc toutes les familles y sont au hasard — c'est le contrôle
négatif de haut de pile. Le bras z n'a pas ce trou, la recalibration 2AFC ayant débloqué
les concepts au bloc 31 (`z2afc_all` y porte 2880 lignes). Le bras alpha de Qwen apporte en prime les blocs 32
et 56, que le balayage en z n'a jamais couverts.

`main` écrit `correct_raw` en booléen là où le moteur actuel écrit 1.0 / 0.0 / 0.5 avec
un drapeau `tie_adjusted` séparé ; `parse_score` lit les deux formes, sans quoi le seul
balayage à plus de quatre concepts serait silencieusement rejeté.

## 9. Ce qui reste ouvert

Sur Llama, les sections 7.3 (au moins 5 concepts, les 32 couches) et 5.7 (autant de
directions aléatoires que de directions conceptuelles) sont désormais satisfaites :
10 × 10 sur les blocs 0–31 en z et 0–30 en alpha. Restent quatre limites.

**Qwen n'a pas été complété.** Il reste à 4 concepts et 3 directions aléatoires, sur
6 couches en z et 8 en alpha. Tout ce que dit la section 6 sur la spécificité du modèle
repose donc, côté Qwen, sur le panel biaisé décrit en section 4. Le refaire coûterait
davantage : le 27B tourne environ quinze fois moins vite que Llama.

**Le bras alpha s'arrête au bloc 30.** `full32_all` n'a jamais balayé le bloc 31 pour la
famille concept, et `--exclude_layers 31` ramène les deux familles au même support plutôt
que de comparer des parts de variance prises sur des ensembles de couches différents.
Combler le trou coûte environ une minute de GPU — 4 concepts × 1 bloc × 10 doses × 40
essais — mais le bloc 31 est un contrôle négatif de haut de pile, où aucune couche
d'attention ne suit l'injection, donc rien d'interprétable n'y est perdu.

**Bruit et dropout restent à 2 réalisations.** La section 5.7 ne demande pas de les
apparier au nombre de concepts, mais leurs moyennes reposent sur deux tirages, contre dix
maintenant pour concept et aléatoire fixe : ce sont les deux familles dont l'estimation
est désormais la moins serrée.

**Les figures de l'expérience 1 ne se régénèrent pas encore sur le panel complet.**
Cartes par couche et heatmaps lisent `summary.json`, et
`merge_experiment1_runs.py` refuse ces runs — « runs disagree on 'families' » — parce
qu'il a été écrit pour recoller des groupes de couches disjoints, pas des runs qui
partagent les couches et diffèrent par les directions. Les chiffres de la section 4
viennent de `experiment1_localization_report.py`, qui met bien les `--run_dir` en commun.
Régénérer les figures demande soit d'étendre le script de fusion, soit de rejouer
`summarize()` sur les essais mis en commun.

## 10. Reproduire

Bras z, panel complet :

```bash
python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/z2afc_all \
    --run_dir results/experiment1/panel-concepts \
    --run_dir results/experiment1/panel-random \
    --matching z --out_dir results/experiment3/llama_z

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/z2afc_qwen38 --matching z \
    --out_dir results/experiment3/qwen_z
```

Bras alpha, panel complet — `--exclude_layers 31` est obligatoire, `full32_all` n'ayant
pas le bloc 31 pour la famille concept :

```bash
python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/full32_all \
    --run_dir results/experiment1/panel-concepts \
    --run_dir results/experiment1/panel-random \
    --matching alpha --exclude_layers 31 \
    --out_dir results/experiment3/llama_alpha
```

Les chiffres par famille de la section 4 :

```bash
python code/analysis/experiment1_localization_report.py \
    --run_dir results/experiment1/z2afc_all \
    --run_dir results/experiment1/panel-concepts \
    --run_dir results/experiment1/panel-random --matching z
```

`--matching` n'est pas optionnel quand on met en commun des balayages d'époques
différentes : sans lui, le bras z inexploitable de `full32_all` se mélangerait aux
chiffres z sans que rien ne le signale.

Bras alpha, depuis les tranches versionnées, qui sont des répertoires de run ordinaires
(`trials.csv` + `summary.json`) :

```bash
for arm in llama_alpha qwen_alpha llama_alpha_10concepts; do
    python code/analysis/experiment3_variability.py \
        --run_dir results/experiment3/${arm}_source --matching alpha \
        --out_dir results/experiment3/$arm
done
```

Vérifié : relancées depuis la tranche versionnée, les analyses alpha redonnent un
`variability_report.json` et un `per_direction_curves.csv` identiques à ceux obtenus
depuis les balayages d'origine. Les tranches ont été produites avec `--export_run_dir`.
