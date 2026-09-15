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
| `llama_z` | Llama-3.1-8B | z | 4 | 0–31 |
| `llama_alpha` | Llama-3.1-8B | alpha | 4 | 0–30 |
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
| `llama_z` | 4,0× | 0,3× |
| `llama_alpha` | 10,4× | 1,3× |
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
| Llama z | concept | 0,088 | 0,600 | 0,014 | 0,298 |
| Llama z | random | 0,011 | 0,656 | 0,011 | 0,322 |
| Llama alpha | concept | 0,102 | 0,544 | 0,018 | 0,336 |
| Llama alpha | random | 0,002 | 0,771 | 0,019 | 0,208 |
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

## 4. Le plan à 10 concepts : une dissociation minoritaire, pas une hétérogénéité générale

`results/experiment1/main` porte les 10 concepts que la section 7.3 réclame. Au bloc 3,
la seule couche vivante de ce balayage (S moyen 0,629 ; les blocs 16 et 28 donnent
−0,004 et 0,009) :

| concept | S au bloc 3 |
|---|---|
| Satellites | 0,868 |
| appreciation | 0,844 |
| betrayal | 0,789 |
| fibonacci_numbers | 0,751 |
| shutdown | 0,731 |
| recursion | 0,703 |
| Origami | 0,679 |
| Illusions | 0,631 |
| **Trumpets** | **0,281** |
| **Dust** | **0,011** |

Huit concepts sur dix tiennent dans 0,631–0,868, avec un écart-type de **0,081** —
c'est-à-dire *plus homogènes entre eux* que ne le sont les 3 directions aléatoires du
même balayage à la même couche (0,338 / 0,568 / 0,618, écart-type 0,149). Deux concepts
décrochent : `Trumpets` à 0,281 et `Dust` à 0,011, soit rien du tout.

C'est le constat de Macar et al. que cite la section 7.2, mais précisé, et dans l'autre
sens que ne le suggérait le panel à 4 concepts : **le corps des concepts se comporte de
façon homogène, et une minorité ne répond pas.** Ce n'est pas « les concepts varient
beaucoup ». Sur le panel à 4 concepts, `Dust` pesait 1 cas sur 4 et gonflait
mécaniquement toute mesure de dispersion ; sur 10 il pèse 1 sur 10 et le noyau devient
visible. L'écart-type entre concepts passe de 0,121 (4 concepts, mêmes couches) à 0,085
(10 concepts) pour cette raison.

## 5. Quel concept décroche dépend du modèle

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

## 6. Pourquoi le bras alpha est ici

La section 7.3 ne demande que des doses en z. Le bras alpha est analysé en plus parce
qu'il est le contrôle qui rend le reste lisible : à
`experiment1_psychometrics.py:852`, sous `matching == "alpha"` l'amplitude *est* la
valeur de grille et `calibration.scale()` n'est jamais consultée. L'appariement alpha ne
dépend donc d'aucune estimation de s(ℓ,v), et la structure de variance qu'il donne ne
peut être ni un artefact de calibration ni un artefact de la plage de la grille z. Elle
reproduit celle du bras z sur Llama : part de direction 0,102 contre 0,088, écart-type
groupé entre concepts 0,135 contre 0,137, et le même concept en dernier.

C'est aussi ce qui rend `main` exploitable, voir la section suivante.

## 7. Provenance des bras alpha

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

## 8. Ce que ce plan ne couvre pas

La section 7.3 demande au moins 5 concepts **et** les 32 couches. Aucun balayage ne
tient les deux à la fois :

- les balayages recalibrés couvrent les 32 couches mais **4 concepts et 3 directions
  aléatoires** ;
- `main` couvre **10 concepts** mais **3 couches** (3, 16, 28), dont une seule vivante.

Le plan à 10 concepts fonde donc la variance entre concepts ; il ne peut fonder ni la
composante de couche ni l'interaction direction × couche, qui viennent des balayages à
4 concepts. Croiser 10 concepts avec les 32 couches demande un balayage neuf : c'est la
seule dépense GPU que l'expérience 3 exigerait encore, et c'est elle qui permettrait de
dire si la dissociation de `Dust` vaut à toute profondeur ou seulement dans la fenêtre
superficielle.

Trois directions aléatoires seulement, dans tous les balayages : leur écart-type repose
sur 3 points et n'est qu'un ordre de grandeur. La section 7.3 en demande autant que de
concepts.

Qwen est lu sur 6 couches en z et 8 en alpha : sa composante de couche est estimée sur
un échantillon de profondeurs, pas sur la profondeur entière.

## 9. Reproduire

Bras z, depuis les balayages recalibrés déjà versionnés :

```bash
python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/z2afc_all --matching z \
    --out_dir results/experiment3/llama_z

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/z2afc_qwen38 --matching z \
    --out_dir results/experiment3/qwen_z
```

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
