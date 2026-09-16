# Expérience 3 — variabilité entre concepts, directions et couches

Doc de cadrage section 7. Les essais sont ceux de l'expérience 1 : le balayage fait
déjà varier `direction_id` à l'intérieur de chaque famille, donc une lecture par
direction des mêmes lignes *est* l'expérience 3, et aucune passe GPU nouvelle n'a été
nécessaire. Ce que `summary.json` ne pouvait pas dire, c'est précisément la question :
il regroupe tous les concepts en une seule courbe `concept` par (couche, appariement).

Analyse : `code/analysis/experiment3_variability.py`.
Résultats : `results/experiment3/`.

| bras | modèle | appariement | familles identifiées | concepts | couches |
|---|---|---|---|---|---|
| `llama_z` | Llama-3.1-8B | z | concept, aléatoire | **10** | 0–31 |
| `llama_alpha` | Llama-3.1-8B | alpha | concept, aléatoire | **10** | 0–30 |
| `llama_alpha_10concepts` | Llama-3.1-8B | alpha | concept, aléatoire | **10** | 3, 16, 28 |
| `qwen_z` | Qwen3.8-27B | z | concept, aléatoire | 4 | 3, 6, 8, 10, 12, 16 |
| `qwen_alpha` | Qwen3.8-27B | alpha | concept, aléatoire | 4 | 3, 6, 8, 10, 12, 16, 32, 56 |
| `qwen_scrambled` | Qwen3.8-27B | alpha | concept, aléatoire, **permuté** | 4 | 3, 6, 12, 32 |

Les cinq bras se lisent chacun sur un seul répertoire groupé versionné de
`results/experiment1/`, donc ils se reproduisent depuis un clone (section 10).

## 1. La mesure

La section 7.5 demande une précision, un contraste et un seuil individuel à 75 %. Le
seuil est presque toujours indisponible ici : la précision 2AFC brute est bloquée près
du hasard par le biais de réponse du modèle, et les courbes *groupées* ne traversent
75 % que dans 0 à 5 couches sur 32. Par direction, une cellule a le quart des essais.
Le seuil est donc rapporté là où la courbe encadre réellement 75 % — 49/768, 22/744,
1/51, 16/66, 34/88 et 11/44 courbes selon le bras, dans l'ordre du tableau ci-dessus —
mais la mesure de travail est le contraste
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
| `qwen_scrambled` | 0,9× | 0,2× |

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
| Qwen permuté | concept | 0,089 | 0,771 | 0,050 | 0,090 |
| Qwen permuté | scrambled | 0,230 | 0,370 | 0,040 | 0,360 |
| Qwen permuté | random | 0,134 | 0,472 | 0,033 | 0,360 |

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

## 7. Le contenu porte le niveau, pas la dispersion

Les sections 5 et 6 établissent que les concepts diffèrent entre eux. Elles ne disent pas
*pourquoi*. Deux lectures restent ouvertes : ou bien la dispersion vient du contenu
sémantique de chaque vecteur, ou bien elle vient de son profil de coordonnées — sa norme,
sa concentration, sa colinéarité avec les directions massives du modèle. Le bras
`qwen_scrambled` sépare les deux.

La famille `scrambled` est le contrôle de contenu de la section 3.3 du cadrage : une
**permutation des coordonnées** du vecteur conceptuel, appariée un à un avec lui
(`prepare_material.py:242`). Elle conserve exactement sa norme et la distribution de ses
coordonnées, et détruit tout alignement avec une direction de feature. À alpha apparié,
une différence entre les deux familles ne peut donc pas être une différence d'amplitude.
C'est le seul bras où la question « le contenu compte-t-il, à magnitude fixée ? » est
posée à l'intérieur de l'expérience 3, sur les mêmes quatre concepts, les mêmes cinq
paires et les mêmes trois doses.

Sur les blocs 3, 6, 12 et 32, S moyen groupé :

| famille | S moyen | écart-type entre directions, groupé | part de variance « direction » |
|---|---|---|---|
| concept | **0,521** | 0,1223 | **0,089** |
| permuté | 0,250 | **0,1489** | **0,230** |
| aléatoire fixe | 0,287 | 0,1394 | 0,134 |

**Le contenu porte le niveau.** Permuter les coordonnées fait tomber S de 0,521 à 0,250,
soit une division par deux à norme strictement identique, et ramène la famille au niveau
des directions aléatoires (0,287). Au bloc 3, la couche la plus vivante, l'écart est le
même : 0,821 pour concept contre 0,509 pour permuté et 0,473 pour aléatoire. C'est la
lecture de l'expérience 1 confirmée sur un contrôle qui neutralise l'amplitude.

**Le contenu ne porte pas la dispersion.** Les vecteurs permutés se dispersent *davantage*
entre eux que les vecteurs intacts — écart-type 0,1489 contre 0,1223, et part de variance
« direction » 0,230 contre 0,089, soit 2,6 fois. Détruire la sémantique n'homogénéise pas
la famille, cela l'hétérogénéise. L'hétérogénéité entre concepts de la section 5 ne peut
donc pas s'expliquer par « certains concepts sont mieux représentés que d'autres » sans
expliquer d'abord pourquoi des vecteurs sans aucun contenu varient encore plus.

**Le classement ne survit pas à la permutation.** Sur les quatre mêmes concepts :

| | intact | permuté |
|---|---|---|
| Satellites | 0,634 (1er) | 0,363 (2e) |
| recursion | 0,597 (2e) | **0,058 (4e)** |
| Dust | 0,490 (3e) | **0,372 (1er)** |
| shutdown | 0,361 (4e) | 0,208 (3e) |

`recursion` passe de deuxième à dernier et `Dust` de troisième à premier. La
détectabilité individuelle d'un concept n'est donc pas portée par le multiensemble de ses
coordonnées : c'est bien son orientation qui la fixe. Mais l'inverse vaut aussi — une
orientation arbitraire suffit à produire une dispersion entre directions du même ordre,
ce qui rejoint la section 2.

Ce bras ne couvre que Qwen, quatre concepts, quatre couches et trois directions
aléatoires. Il tranche la question de l'amplitude ; il ne remplace pas un panel complet.

**Les chiffres de cette section ne se comparent pas à ceux de `qwen_alpha`.** Les deux
bras viennent de balayages différents et la règle de la section 1 leur donne des doses
différentes — [32, 64, 128] ici, [16, 32, 64] pour `qwen_alpha`, qui a un seuil groupé
rapportable. Le S moyen de la famille concept vaut donc 0,821 au bloc 3 ici et 1,065
là-bas, sans que rien ait changé dans la mesure. Les trois familles de cette section
partagent en revanche leurs doses, leurs couches et leurs paires, ce qui est tout ce dont
la comparaison a besoin.

## 8. Pourquoi le bras alpha est ici

La section 7.3 ne demande que des doses en z. Le bras alpha est analysé en plus parce
qu'il est le contrôle qui rend le reste lisible : à
`experiment1_psychometrics.py:852`, sous `matching == "alpha"` l'amplitude *est* la
valeur de grille et `calibration.scale()` n'est jamais consultée. L'appariement alpha ne
dépend donc d'aucune estimation de s(ℓ,v), et la structure de variance qu'il donne ne
peut être ni un artefact de calibration ni un artefact de la plage de la grille z. Elle
reproduit celle du bras z sur Llama : part de direction 0,102 contre 0,088, écart-type
groupé entre concepts 0,135 contre 0,137, et le même concept en dernier.

C'est aussi ce qui rend `main` exploitable, voir la section suivante.

## 9. Provenance des bras alpha

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

Seules les lignes alpha de ces balayages ont été extraites ; leur bras z défectueux reste
délibérément hors du dépôt. Les extractions ont d'abord été versionnées comme répertoires
de run à part entière (`results/experiment3/*_source/`), à une époque où aucun balayage
source n'était dans le dépôt. Depuis que les panels groupés de `results/experiment1` le
sont, `llama_alpha_source` et `qwen_alpha_source` ne sont plus que des copies exactes de
`llama_alpha_panel` et de `qwen_alpha_panel` — vérifié ligne à ligne, 297 600 et 38 720
essais, aucune clé manquante et aucun contraste différent — et les cinq bras lisent
désormais les panels, pas ces copies. `llama_alpha_10concepts_source`, lui, reste
nécessaire : le balayage `main` dont il vient n'est nulle part ailleurs dans le dépôt.

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

## 10. Ce qui reste ouvert

Sur Llama, les sections 7.3 (au moins 5 concepts, les 32 couches) et 5.7 (autant de
directions aléatoires que de directions conceptuelles) sont désormais satisfaites :
10 × 10 sur les blocs 0–31 en z et 0–30 en alpha. Le contrôle de contenu de la
section 3.3 est en place, sur Qwen. Restent cinq limites.

**Qwen n'a pas été complété.** Il reste à 4 concepts et 3 directions aléatoires, sur
6 couches en z, 8 en alpha et 4 pour le bras permuté. Tout ce que disent les sections 6
et 7 sur la spécificité du modèle repose donc, côté Qwen, sur le panel biaisé décrit en
section 4. Le refaire coûterait
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
est désormais la moins serrée. Elles n'entrent pas dans la décomposition, qui ne porte que
sur les familles à direction identifiée — concept, aléatoire fixe, permuté — puisque bruit
et dropout retirent leur perturbation à chaque essai et qu'un écart entre leurs
« directions » serait du bruit d'échantillonnage, pas de la variance entre directions.

**Le contrôle de contenu ne couvre que Qwen, et seulement en alpha.** La section 7 repose
sur 4 concepts, 4 blocs et 3 directions aléatoires. Les vecteurs permutés ne sont produits
que si la calibration active `directions.scrambled_concept`, ce que celle de Llama ne fait
pas : la question « le contenu compte-t-il à magnitude fixée ? » n'a donc pas de réponse
sur le modèle principal, et c'est le trou le plus utile à combler ensuite.

**Les figures de l'expérience 1 n'ont pas encore été régénérées sur le panel complet**
— mais le blocage, lui, est levé. Cartes par couche et heatmaps lisent `summary.json`, et
`merge_experiment1_runs.py` refusait ces runs — « runs disagree on 'families' » — parce
qu'il a été écrit pour recoller des groupes de couches disjoints, pas des runs qui
partagent les couches et diffèrent par les directions. Il n'est plus dans le chemin :
`consolidate_experiment1_panel.py` a déjà rejoué `summarize()` sur les essais mis en
commun, donc chaque panel porte un `summary.json` recalculé et se donne tel quel au script
de figures. Un panel ne portant qu'un bras, il faut le nommer, faute de quoi le script
cherche l'autre grille de doses et s'arrête sur `KeyError: 'alpha'` :

```bash
python code/analysis/plot_experiment1_layer_maps.py \
    --run_dir results/experiment1/llama_z_panel --matchings z
python code/analysis/plot_experiment1_layer_maps.py \
    --run_dir results/experiment1/llama_alpha_panel --matchings alpha
```

Vérifié le 2026-09-16 : les cinq figures sortent pour chaque bras. Reste à décider quelles
figures du manuscrit sont remplacées, ce qui relève de l'expérience 1. Les chiffres de la
section 4 viennent de `experiment1_localization_report.py`, qui met bien les `--run_dir`
en commun.

## 11. Reproduire

Depuis 2026-09-16, seuls cinq répertoires groupés de `results/experiment1` sont versionnés
(`llama_z_panel`, `llama_alpha_panel`, `qwen_z_panel`, `qwen_alpha_panel`,
`qwen_scrambled_panel`) ; les balayages sources restent locaux. Chaque panel porte déjà le
bon bras, donc un seul `--run_dir` suffit et `--matching` ne fait plus que confirmer :

```bash
python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/llama_z_panel --matching z \
    --out_dir results/experiment3/llama_z

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/llama_alpha_panel --matching alpha \
    --out_dir results/experiment3/llama_alpha

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/qwen_z_panel --matching z \
    --out_dir results/experiment3/qwen_z

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/qwen_alpha_panel --matching alpha \
    --out_dir results/experiment3/qwen_alpha

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/qwen_scrambled_panel --matching alpha \
    --out_dir results/experiment3/qwen_scrambled
```

`llama_alpha_panel` exclut déjà le bloc 31, donc `--exclude_layers` n'est plus nécessaire.
Vérifié le 2026-09-16 : les quatre bras qui existaient déjà redonnent, depuis les panels,
des rapports strictement identiques à ceux obtenus depuis les balayages sources — mêmes
composantes de variance, `per_direction_curves.csv` et figures inchangés octet pour octet.
Seuls `runs` et `excluded_layers` changent, parce que la provenance est désormais un
répertoire versionné et que le panel alpha de Llama porte déjà son exclusion du bloc 31.

Les chiffres par famille de la section 4 :

```bash
python code/analysis/experiment1_localization_report.py \
    --run_dir results/experiment1/llama_z_panel --matching z
```

Reconstruire un panel groupé à partir des balayages sources, si on les a localement :

```bash
python code/analysis/consolidate_experiment1_panel.py \
    --run_dir results/experiment1/z2afc_all \
    --run_dir results/experiment1/panel-concepts \
    --run_dir results/experiment1/panel-random \
    --matching z --out results/experiment1/llama_z_panel
```
