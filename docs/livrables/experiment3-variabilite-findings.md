# Expérience 3 — variabilité entre concepts, directions et couches

Doc de cadrage section 7. Les essais sont ceux de l'expérience 1 : le balayage fait
déjà varier `direction_id` à l'intérieur de chaque famille, donc une lecture par
direction des mêmes lignes *est* l'expérience 3, et aucune passe GPU nouvelle n'a été
nécessaire. Ce que `summary.json` ne pouvait pas dire, c'est précisément la question :
il regroupe tous les concepts en une seule courbe `concept` par (couche, appariement).

Analyse : `code/analysis/experiment3_variability.py`.
Résultats : `results/experiment3/{llama,qwen}_{z,alpha}/`.

## 1. La mesure

La section 7.5 demande une précision, un contraste et un seuil individuel à 75 %. Le
seuil est presque toujours indisponible ici : la précision 2AFC brute est bloquée près
du hasard par le biais de réponse du modèle, et les courbes *groupées* ne traversent
75 % que dans 0 à 5 couches sur 32. Par direction, une cellule a le quart des essais.
Le seuil est donc rapporté là où la courbe encadre réellement 75 % — 26/352, 11/348,
16/66 et 34/88 courbes selon le bras — mais la mesure de travail est le contraste de
localisation apparié de la section 5.8,

    S = (contraste quand A est ciblée − contraste quand B est ciblée) / 2

sur les deux essais qui partagent tout sauf la phrase touchée. Tout décalage
indépendant de la cible s'annule, ce qui rend S lisible là où la précision ne l'est
pas. C'est l'estimateur et la clé d'appariement de `experiment1_localization_report.py`.

Les trois doses de la section 7.3 sont fixées avant la lecture par direction, à partir
de la courbe groupée : le seuil global quand il est rapportable (bras alpha de Qwen,
32,2), sinon la dose testée où |S| groupé est maximal. Ce sont toujours des valeurs de
la grille, jamais interpolées.

## 2. Composantes de variance

Décomposition en moyennes sur les cellules (direction × couche × phrase), qui est la
partie fixe du modèle mixte de la section 7.6 : le plan est équilibré par construction,
donc les termes sont orthogonaux et les parts somment à 1 sans ajustement.

| bras | famille | direction | couche | phrase | interaction |
|---|---|---|---|---|---|
| Llama z | concept | **0,088** | 0,600 | 0,014 | 0,298 |
| Llama z | random | 0,011 | 0,656 | 0,011 | 0,322 |
| Llama alpha | concept | **0,102** | 0,544 | 0,018 | 0,336 |
| Llama alpha | random | 0,002 | 0,774 | 0,018 | 0,207 |
| Qwen z | concept | **0,288** | 0,357 | 0,092 | 0,263 |
| Qwen z | random | 0,062 | 0,295 | 0,071 | 0,572 |
| Qwen alpha | concept | **0,078** | 0,810 | 0,020 | 0,092 |
| Qwen alpha | random | 0,013 | 0,697 | 0,013 | 0,276 |

Trois lectures.

**La couche domine partout** (0,36 à 0,81). L'effet moyen de l'expérience 1 n'est pas
homogène en profondeur, ce qui était déjà visible sur les cartes par couche ; ce qui est
nouveau, c'est que la profondeur pèse plus que l'identité de la direction dans tous les
bras sauf un.

**L'identité du concept porte 8 à 29 % de la variance, celle d'une direction aléatoire
0,2 à 6 %.** L'écart-type entre concepts vaut 0,127 à 0,193 contre 0,013 à 0,034 entre
directions aléatoires, soit un facteur 4 à 11. Les concepts diffèrent donc entre eux
bien plus que des directions tirées au hasard ne diffèrent entre elles : la dispersion
n'est pas du bruit d'échantillonnage de direction. C'est le constat de Macar et al. que
la section 7.2 cite.

**La phrase ne porte presque rien** (0,011 à 0,092). Les 5 paires du corpus ne sont pas
une source de variance sur cette tâche.

**L'interaction direction × couche est substantielle** (0,092 à 0,572). La mise en garde
de la section 7.7 s'applique : une différence entre familles ne peut pas être attribuée
à l'architecture sans nommer le site d'injection.

## 3. Quel concept est facile dépend du modèle

S moyen par concept, aux trois doses retenues :

| concept | Llama z | Llama alpha | Qwen z | Qwen alpha |
|---|---|---|---|---|
| Dust | **0,063** | **0,043** | 0,524 | 0,352 |
| Satellites | 0,348 | 0,327 | **0,779** | **0,520** |
| recursion | 0,308 | 0,297 | 0,610 | 0,518 |
| shutdown | 0,350 | 0,312 | **0,316** | **0,265** |

À l'intérieur d'un modèle, les deux appariements donnent le même ordre — sur Qwen il est
identique, sur Llama seuls `Satellites` et `shutdown` s'échangent, et ils sont à 0,02
l'un de l'autre. Entre modèles, l'ordre se retourne : `Dust` est le concept nettement le
plus faible sur Llama (0,063 / 0,043) mais milieu de tableau sur Qwen, et `shutdown` est
parmi les plus forts sur Llama alors qu'il est le plus faible sur Qwen.

La variabilité entre concepts est donc réelle et reproductible sous les deux variables de
dose, mais *quel* concept est détecté est une propriété du modèle, pas du concept. Cela
renforce le constat de la section 7.2 : non seulement la norme du vecteur ne le prédit
pas, mais l'identité du concept ne se transporte pas d'un modèle à l'autre.

## 4. Pourquoi le bras alpha est ici

La section 7.3 ne demande que des doses en z. Le bras alpha est analysé en plus parce
qu'il est le contrôle qui rend la section 2 lisible : à
`experiment1_psychometrics.py:852`, sous `matching == "alpha"` l'amplitude *est* la
valeur de grille et `calibration.scale()` n'est jamais consultée. L'appariement alpha ne
dépend donc d'aucune estimation de s(ℓ,v), et la structure de variance qu'il donne ne
peut pas être un artefact de la calibration ni de la plage de la grille z. Elle
reproduit celle du bras z : part de direction 0,102 contre 0,088 sur Llama, écart-type
entre concepts 0,135 contre 0,137, et le même ordre des concepts. C'est ce qui autorise
à lire la section 2 comme de la géométrie et non comme un effet de la variable de dose.

Corollaire de provenance : les lignes alpha proviennent de balayages antérieurs à la
recalibration 2AFC (`full32_all`, `qwen38_all_layers`). Le défaut de calibration corrigé
le 14/09 n'atteignait que le bras z de ces runs ; leur bras alpha est valide tel quel.
Seules les lignes alpha ont été extraites et versionnées, comme
répertoires de run à part entière (`results/experiment3/{llama,qwen}_alpha_source/`), le
bras z défectueux de ces runs restant délibérément hors du dépôt. Le bras alpha de Qwen apporte en prime les blocs 32 et 56,
que le balayage en z n'a jamais couverts.

## 5. Ce que ce plan ne couvre pas

La section 7.3 demande **au moins 5 concepts et autant de directions aléatoires**. Les
balayages recalibrés portent **4 concepts et 3 directions aléatoires**. Les composantes
ci-dessus sont donc calculées sur un panel plus étroit que le plan, et l'écart-type entre
concepts repose sur 4 points : il indique un ordre de grandeur, pas une estimation serrée.

Le seul run à 10 concepts (`results/experiment1/main` — `appreciation`, `betrayal`,
`Dust`, `fibonacci_numbers`, `Illusions`, `Origami`, `recursion`, `Satellites`,
`shutdown`, `Trumpets`) ne couvre que 3 couches (3, 16, 28) et ne porte pas de
`calibration_dir`. Il satisfait le minimum de concepts mais pas la condition « les 32
couches », et ne peut pas fonder la composante de couche. Élargir le panel à 5 concepts
ou plus sur les 32 couches demande un balayage neuf ; c'est la seule dépense GPU que
l'expérience 3 exigerait encore.

Qwen est lu sur 6 couches en z (3, 6, 8, 10, 12, 16) et 8 en alpha (plus 32 et 56) : sa
composante de couche est estimée sur un échantillon de profondeurs, pas sur la
profondeur entière.

## 6. Reproduire

Les bras z partent des balayages recalibrés, déjà versionnés :

```bash
python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/z2afc_all --matching z \
    --out_dir results/experiment3/llama_z

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment1/z2afc_qwen38 --matching z \
    --out_dir results/experiment3/qwen_z
```

Les bras alpha partent de la tranche versionnée, qui est un répertoire de run ordinaire
(`trials.csv` + `summary.json`) :

```bash
python code/analysis/experiment3_variability.py \
    --run_dir results/experiment3/llama_alpha_source --matching alpha \
    --out_dir results/experiment3/llama_alpha

python code/analysis/experiment3_variability.py \
    --run_dir results/experiment3/qwen_alpha_source --matching alpha \
    --out_dir results/experiment3/qwen_alpha
```

Vérifié : relancées depuis la tranche versionnée, les deux analyses alpha redonnent un
`variability_report.json` et un `per_direction_curves.csv` identiques à ceux obtenus
depuis `full32_all` et `qwen38_all_layers`. Les tranches ont été produites avec
`--export_run_dir`.
