# Cadrage expérimental

## Psychophysique de la détection des perturbations internes dans les LLM

### Introspection, anomalie ou géométrie ?

## 1. Question de recherche

Des travaux récents montrent que certains modèles de langage peuvent rapporter qu'une intervention a modifié leurs activations. Ces résultats restent difficiles à comparer. Les études emploient des perturbations, des échelles d'intensité, des tâches et des formats de réponse différents. Une bonne performance peut alors correspondre à plusieurs phénomènes : détection d'un contenu conceptuel, détection d'une activation inhabituelle, biais de réponse ou simple dysfonctionnement du modèle.

La question principale est la suivante :

> Lorsqu'on compare des perturbations conceptuelles, des directions aléatoires fixes et du bruit, les différences de détectabilité restent-elles les mêmes à amplitude brute appariée et à amplitude rapportée aux variations naturelles des activations ?

Le projet construit un protocole de psychophysique computationnelle autour de cette question. Il ne suppose pas qu'un auto-rapport constitue une preuve de conscience ou de métacognition. Il mesure une discrimination comportementale et cherche à déterminer ce qui l'explique.

La contribution principale porte sur les courbes psychométriques obtenues sous deux paramétrisations de l'intensité :

- une amplitude brute appariée, notée $\alpha$
- une amplitude standardisée appariée, notée $z=\alpha/s(\ell,v)$.

Ici, $\ell$ désigne la couche, $v$ une direction de norme unitaire et $s(\ell,v)$ l'échelle de variabilité des activations sans intervention projetées dans cette direction. L'amplitude $\alpha$ mesure la norme du vecteur ajouté à chaque token ciblé, et $z$ exprime cette amplitude relativement à l'échelle naturelle, supposée strictement positive. Ces deux doses sont distinctes de l'énergie totale d'une intervention sur plusieurs tokens.

La localisation 2AFC sert d'instrument de mesure et de réplication minimale. Les essais sham, la détection de présence, la mesure du critère de réponse et une tâche témoin complètent ce dispositif.

## 2. Hypothèses

| Identifiant | Hypothèse | Expérience principale |
| --- | --- | --- |
| H1 | Déplacer l'injection de A vers B déplace la préférence du modèle vers la phrase effectivement ciblée. | Expérience 1 |
| H2 | Le modèle distingue une intervention réelle d'un sham au-delà de son biais de réponse. | Expérience 2 |
| H3 | À $\alpha$ apparié, les courbes de détectabilité diffèrent entre concept, direction aléatoire fixe et bruit. | Expérience 3 |
| H4 | L'appariement en $z$ modifie les différences observées à $\alpha$ apparié. | Expérience 4 |
| H5 | Les seuils de détection varient entre concepts, directions et couches. | Expérience 5 |
| H6 | La détection reste mesurable dans une plage où la performance de la tâche témoin est préservée. | Expérience 6 |
| H7 | Un texte évocateur du concept modifie le taux de fausses déclarations d’injection interne. | Expérience 7 |
| H8 | La confiance apporte une information que le choix discret ne résume pas entièrement. | Expérience 8 |
| H9 | Le modèle estime le nombre d'injections au-delà de ce qu'explique leur intensité totale. | Expérience 9 |
| H10 | Le modèle identifie le contenu de deux injections distinctes. | Expérience 10 |
| H11 | Le modèle rapporte l'ordre relatif des couches de deux injections. | Expérience 11 |

Les expériences 0 à 8 forment le cadrage principal. Les expériences 9 à 11 étendent le protocole à la multi-injection. Elles ne doivent pas retarder l'estimation des courbes psychométriques principales.

## 3. Méthode commune

### 3.1 Modèle et exécution

Le modèle principal est `meta-llama/Llama-3.1-8B-Instruct`. D'autres modèles tel que `Qwen/Qwen3.8-27B` pourront être testé une fois les protocoles validés avec le modèle Llama. 

### 3.2 Données

Le corpus principal comprend les 100 phrases distinctes de `LOCALIZATION_SENTENCES`, dans `code/utils/all_prompts.py`, utilisées par le script de détection du dépôt. Les mêmes phrases servent à la calibration de l'échelle naturelle et à l'évaluation comportementale des expériences de détection. La calibration utilise uniquement les activations sans intervention, sans utiliser les résultats de détection. L'échelle est figée avant l'évaluation. Un corpus indépendant pourra être utilisé ensuite pour évaluer la généralisation de cette calibration et des résultats.

Les phrases sont appariées en longueur de tokens autant que possible. Un séparateur fixe est utilisé. Les positions exactes des phrases, le nombre de tokens qui les séparent et la distance entre les sites d'injection et le token de réponse sont enregistrés.

Chaque expérience, de l'expérience 0 à l'expérience 11, précise dans sa propre section les données utilisées et les valeurs de son plan : nombre de phrases ou de paires, nombre et identité des concepts, nombre de directions aléatoires, nombre de réalisations de bruit, couches explorées, doses et nombre d'essais par condition. Les paramètres sans objet pour une expérience sont indiqués comme tels. Les valeurs encore indéterminées sont signalées et accompagnées de la méthode prévue pour les fixer avant l'exécution.

Les expériences 3 et 4 utilisent les mêmes paires de phrases, directions, couches et règles de présentation afin de comparer les deux règles d'appariement de l'intensité. Leurs effectifs sont explicités dans leurs sections respectives.

### 3.3 Construction des directions conceptuelles

Deux méthodes construisent les directions conceptuelles, selon les données simples ou complexes de Hahami et al. [1]. Dans les deux cas, $c$ désigne le concept et $\ell$ la couche d'extraction. On note $a_\ell(u)$ le vecteur d'activation extrait du prompt construit à partir du mot ou de la phrase $u$, sans intervention. Ce vecteur utilise soit l'activation au dernier token du prompt, soit la moyenne des activations sur tous ses tokens. Chaque expérience précise la variante retenue et l'utilise pour le concept comme pour ses témoins.

#### 3.3.1 Concepts simples

Le fichier `data/dataset/simple_data.json` contient cinq concepts : `Dust`, `Satellites`, `Trumpets`, `Origami` et `Illusions`, ainsi qu'une liste de mots témoins. Le prompt de construction est `Tell me about {WORD}.`, mis au format de conversation du modèle.

Pour chaque concept, on soustrait à son activation la moyenne des activations des mots témoins :

$$
\tilde v_{c,\ell}
=
a_\ell(c)
-\frac{1}{|B|}\sum_{b\in B}a_\ell(b).
$$

Ici, $\tilde v_{c,\ell}$ est le vecteur conceptuel brut, $B$ la liste des mots témoins utilisés, $|B|$ son nombre d'entrées et $b$ une entrée de cette liste. La somme calcule leur activation moyenne. Le code actuel utilise les 50 premières entrées de la liste de mots témoins. Chaque expérience précise les concepts sélectionnés et les mots témoins utilisés.

#### 3.3.2 Concepts complexes

Le fichier `data/dataset/complex_data.json` décrit chaque concept par des phrases positives et des phrases négatives ou contrastantes. Chaque phrase est présentée dans le format de conversation du modèle.

La direction brute est la différence entre les deux moyennes d'activations :

$$
\tilde v_{c,\ell}
=
\frac{1}{|P_c|}\sum_{u\in P_c}a_\ell(u)
-
\frac{1}{|N_c|}\sum_{u\in N_c}a_\ell(u).
$$

Ici, $P_c$ et $N_c$ désignent respectivement les ensembles de phrases positives et négatives du concept $c$. Leurs tailles sont $|P_c|$ et $|N_c|$, et $u$ désigne une phrase de l'ensemble concerné. Chaque expérience précise les concepts retenus et le nombre de phrases utilisées dans chaque ensemble.

#### 3.3.3 Normalisation commune

Pour les deux méthodes, le vecteur brut non nul est normalisé avant l'injection :

$$
v_{c,\ell}
=
\frac{\tilde v_{c,\ell}}
{\lVert \tilde v_{c,\ell}\rVert_2}.
$$

Ici, $\lVert\tilde v_{c,\ell}\rVert_2$ est la norme euclidienne du vecteur brut et $v_{c,\ell}$ la direction de norme unitaire obtenue. Un vecteur brut nul ne peut pas être normalisé et est exclu. La méthode de construction et la variante d'extraction sont précisées pour chaque direction utilisée.

Les exemples utilisés pour construire une direction ne sont pas employés pour évaluer cette même direction. Les concepts sont répartis entre un ensemble de mise au point et un ensemble *hold-out*.

### 3.4 Directions aléatoires fixes

Pour chaque couche, une banque de vecteurs est tirée avec une graine enregistrée. Chaque vecteur est normalisé à l'unité et reste identique pour tous les essais qui portent son identifiant :

$$
r_{k,\ell}\sim\mathcal N(0,I),
\qquad
v_{k,\ell}
=
\frac{r_{k,\ell}}{\lVert r_{k,\ell}\rVert_2}.
$$

Ici, $k$ identifie une direction de la banque et $r_{k,\ell}$ est le vecteur tiré à la couche $\ell$, de dimension égale à celle des activations. La notation $r_{k,\ell}\sim\mathcal N(0,I)$ signifie que ses composantes sont des variables gaussiennes indépendantes, de moyenne zéro et de variance un. $I$ désigne la matrice identité de cette dimension. La division par la norme euclidienne $\lVert r_{k,\ell}\rVert_2$ donne la direction unitaire $v_{k,\ell}$.

Ce contrôle teste l'effet d'un déplacement cohérent dans une direction sans contenu conceptuel construit.

### 3.5 Bruit à direction renouvelée

Le bruit se distingue de la direction aléatoire fixe parce que sa direction est renouvelée pour chaque token ciblé. Toutes les réalisations sont tirées et enregistrées avant l'exécution. Pour le token $t$ d'un essai $q$ :
$$
g_{q,t}\sim\mathcal N(0,I),
\qquad
v_{q,t}=\frac{g_{q,t}}{\lVert g_{q,t}\rVert_2}.
$$

Ici, $q$ identifie l'essai et $t$ le token ciblé. Le vecteur $g_{q,t}$ a la même dimension que les activations et suit la loi gaussienne définie en section 3.4. Sa division par sa norme euclidienne donne la direction unitaire $v_{q,t}$. On tire donc une nouvelle direction pour chaque token de chaque essai, puis on règle séparément l'amplitude injectée. Ces tirages concernent une couche d'injection $\ell$ donnée, dont l'indice est omis dans cette équation.

Pour l'appariement en $\alpha$, on applique $\Delta h_{q,t}=\alpha v_{q,t}$. Pour l'appariement en $z$, on applique $\Delta h_{q,t}=z\,s(\ell,v_{q,t})v_{q,t}$. Le symbole $\Delta h_{q,t}$ désigne le vecteur ajouté aux activations du token $t$ de l'essai $q$, à la couche d'injection. La notation abrégée $\Delta h$ désigne cette modification lorsque les indices ne sont pas nécessaires.

### 3.6 Site et positions d'injection

Toutes les familles de perturbations sont appliquées au même hook. Le hook correspond à la sortie de la couche choisie. L'intervention vise tous les tokens de la phrase ciblée, sauf les tokens de contrôle explicitement exclus avant l'expérience.

Pour le modèle principal, les couches sont les 32 blocs Transformer, indexés de 0 à 31. L'indice désigne le bloc dont on utilise la sortie, pour la calibration comme pour l'injection. Les expériences à injection unique balayent les 32 couches, avec une seule couche perturbée par essai. Chaque expérience à plusieurs injections précise les combinaisons de couches testées.

## 4. Expérience 0 : estimation de l'échelle naturelle $s(\ell,v)$

### 4.1 Hypothèse méthodologique

Les activations naturelles ne présentent pas la même variabilité selon la couche et la direction. Deux perturbations de même amplitude $\alpha$ peuvent donc correspondre à des valeurs de $z$ différentes.

### 4.2 Objectif

Estimer $s(\ell,v)$ à partir des activations sans intervention des mêmes 100 phrases que celles utilisées pour l'évaluation, sans utiliser les résultats de détection. Comparer SD et MAD, puis construire les grilles de $\alpha$ et de $z$ des expériences comportementales. L'échelle est figée avant la mesure des performances de détection.

### 4.3 Matériel

- les 100 phrases distinctes de `LOCALIZATION_SENTENCES`, également utilisées pour l'évaluation comportementale
- les positions de tokens admissibles dans les futures expériences
- toutes les directions conceptuelles et aléatoires prévues
- les directions aléatoires de bruit prévues pour chaque essai et chaque token ciblé, aux couches explorées. Elles sont tirées avant la calibration afin d'estimer leur échelle naturelle, puis réutilisées dans les comparaisons à amplitude brute et standardisée. Les vecteurs ou les graines permettant de les régénérer sont conservés
- les 32 couches du modèle principal, indexées de 0 à 31. La calibration porte sur la sortie de chaque bloc Transformer, au même site que l'injection

Les 100 phrases et les 32 couches sont retenues. Le nombre de contextes de présentation et de positions extraites, ainsi que la liste et le nombre des directions conceptuelles, aléatoires et de bruit, seront fixés avec les configurations des expériences qu'elles calibrent. Aucune dose d'injection ni réponse de détection n'est utilisée pour estimer cette échelle. Le nombre de passages dépend des contextes retenus, un passage pouvant fournir les activations de toutes les couches.

### 4.4 Déroulé

1. Passer chaque phrase témoin dans le modèle sans intervention.
2. Extraire $h_{\ell,t}$ au hook et aux positions admissibles.
3. Calculer $p_{x,t}(\ell,v)=\langle h_{\ell,t}(x),v\rangle$ pour chaque direction.
4. Estimer la moyenne, l'écart-type, la médiane et la MAD des projections.
5. Tracer la distribution des projections par couche et par famille.
6. Vérifier la stabilité de $s$ par bootstrap des phrases.
7. Figer $s_{\mathrm{SD}}$ avant les expériences 3 et 4.
8. Fournir les échelles nécessaires pour convertir les doses standardisées du pilote comportemental de la section 17.6 en coefficients $\alpha=z\,s(\ell,v)$.

Dans la projection de l'étape 3, $x$ est le texte présenté dans le contexte du prompt retenu et $h_{\ell,t}(x)$ est son vecteur d'activation au token $t$, à la sortie du bloc $\ell$, sans intervention. Les crochets $\langle\cdot,\cdot\rangle$ désignent le produit scalaire. Le résultat $p_{x,t}(\ell,v)$ est un nombre qui mesure la composante de cette activation dans la direction $v$. Le contexte et les positions utilisés pour cette estimation sont conservés avec la calibration.

### 4.5 Données enregistrées

Pour chaque couche et chaque direction : nombre de phrases, nombre de positions, moyenne, SD, MAD corrigée, quantiles, intervalle bootstrap et graine de construction.

### 4.6 Analyse

L'analyse principale utilise la SD. L'analyse de sensibilité recalcule les doses avec la MAD. Le rapport $s_{\mathrm{SD}}/s_{\mathrm{MAD}}$ sert à repérer les distributions à queues lourdes ou sensibles aux valeurs extrêmes.

La SD est l'écart-type des projections. La MAD est la médiane de leurs écarts absolus à leur médiane. La MAD corrigée applique un facteur de correction pour fournir une échelle comparable à la SD sous une distribution gaussienne. Les notations $s_{\mathrm{SD}}$ et $s_{\mathrm{MAD}}$ désignent ces deux estimations de $s(\ell,v)$, pour la même couche et la même direction. Le facteur de correction et la règle de pondération des tokens et phrases seront explicités dans la configuration de calibration.

### 4.7 Résultats interprétables

- Des $s(\ell,v)$ très différents justifient une comparaison à $z$ apparié.
- Des résultats stables entre SD et MAD renforcent la robustesse de la normalisation.
- Une instabilité importante impose d'élargir le corpus témoin ou de limiter les conclusions aux directions dont $s$ est estimé avec précision.

## 5. Expérience 1 : validation de la localisation 2AFC

### 5.1 Hypothèse H1

Déplacer l'injection de la phrase A vers la phrase B déplace le score du modèle vers la phrase effectivement ciblée.

### 5.2 Objectif

Reproduire un résultat canonique de localisation et vérifier que le pipeline d'injection, les positions de tokens et la lecture des logits fonctionnent avant la comparaison des familles.

### 5.3 Prompt principal

```text
User:
You will read two sentences. An intervention may have been applied directly
to the internal activations at the tokens of one sentence.

A) {SENTENCE_A}
B) {SENTENCE_B}

If an intervention occurred, which sentence was directly targeted?
Make your best forced choice.
Answer with exactly one letter: A or B.

Assistant: The answer is
```

Une variante échangera les étiquettes visibles des positions physiques afin de distinguer une préférence de position d'une préférence pour la lettre A ou B. 

### 5.4 Matrice des essais

Pour chaque paire $(x,y)$, où $x$ et $y$ sont les deux phrases :

| Ordre | Injection en A | Injection en B | Sham |
| --- | --- | --- | --- |
| A = x, B = y | 1 essai | 1 essai | 1 essai |
| A = y, B = x | 1 essai | 1 essai | 1 essai |

Le bloc complet contient donc six essais appariés par paire, couche, direction et dose.

Le corpus comprend les 100 phrases communes et les 32 couches sont explorées. Le nombre de paires retenues, les concepts, les variantes de vecteurs et la grille de doses restent à fixer pour la validation initiale avec le pilote de la section 17.6. Les directions aléatoires et le bruit sont comparés dans les expériences 3 et 4. Les six essais du tableau incluent deux sham, réutilisables entre configurations au prompt et au calcul sans modification identiques. Le total des essais est calculé après fixation de ces paramètres et des permutations supplémentaires d'étiquettes.

### 5.5 Variables

- Variables manipulées : phrase ciblée, ordre des phrases, dose et couche.
- Mesure des logits A et B au premier token de réponse.
- Mesure continue :

$$
L=\operatorname{logit}(A)-\operatorname{logit}(B).
$$

Ici, $\operatorname{logit}(A)$ et $\operatorname{logit}(B)$ désignent les scores de sortie non normalisés des tokens de réponse A et B, au premier token de réponse. Le contraste $L$ est positif lorsque A est préféré à B, et négatif dans le cas contraire.

### 5.6 Déroulé

1. Sélectionner deux phrases de longueurs proches.
2. Construire les deux ordres.
3. Repérer les tokens exacts des phrases.
4. Exécuter un sham pour chaque ordre.
5. Injecter la direction dans les tokens de A, puis dans ceux de B.
6. Extraire les logits de A et B au premier token de réponse.
7. Répéter pour toutes les doses et couches prévues.

### 5.7 Scores

Le score ajusté retire le biais propre au même ordre :

$$
L_{\mathrm{ajusté}}
=
L_{\mathrm{injecté}}-L_{\mathrm{sham}}.
$$

Les indices « injecté » et « sham » désignent les deux conditions du même prompt. Le score $L_{\mathrm{ajusté}}$ mesure donc le déplacement du contraste A/B par rapport à la référence sans modification des activations.

Le contraste de localisation est :

$$
S
=
\frac{1}{2}
\left(
L_{A\,\mathrm{perturbé}}
-
L_{B\,\mathrm{perturbé}}
\right).
$$

Ici, $L_{A\,\mathrm{perturbé}}$ est le contraste entre les logits de réponse A et B lorsque l'injection cible A, et $L_{B\,\mathrm{perturbé}}$ le même contraste lorsque l'injection cible B. Les deux essais utilisent les mêmes phrases, le même ordre, la même couche, la même direction et la même dose. Le facteur $1/2$ exprime la demi-différence entre ces deux conditions.

Un contraste $S>0$ signifie que le modèle préfère davantage A lorsque l'injection cible A que lorsqu'elle cible B. Un contraste nul indique l'absence de différence entre ces deux conditions, et un contraste négatif indique un déplacement dans le sens opposé à celui attendu. Ce contraste complète l'accuracy, sans la remplacer : un modèle peut répondre A dans les deux essais tout en présentant un contraste positif. Si les étiquettes visibles sont permutées, la cible et les logits sont identifiés selon les étiquettes effectivement présentées.

### 5.8 Mesures rapportées

- accuracy brute
- accuracy calculée avec le logit ajusté
- contraste moyen $S$ et intervalle de confiance
- biais A/B sur les sham
- taux de réponses invalides
- seuil à 75 % si plusieurs doses couvrent la transition psychométrique.

### 5.9 Diagnostic de contamination de la première phrase vers la seconde

Quand la première phrase est perturbée, on compare les activations des tokens de la seconde phrase à celles du sham :

$$
P_{1\rightarrow2}(\ell')
=
\frac{
\lVert H^{(1)}_{\ell',2}-H^{(0)}_{\ell',2}\rVert_F
}{
\lVert H^{(0)}_{\ell',2}\rVert_F+\varepsilon
}.
$$

Ici, $\ell'$ désigne la couche où le diagnostic est mesuré. Les matrices $H^{(1)}_{\ell',2}$ et $H^{(0)}_{\ell',2}$ contiennent les activations des tokens de la seconde phrase, respectivement avec une injection dans la première phrase et dans le sham. La norme de Frobenius $\lVert\cdot\rVert_F$ est la racine carrée de la somme des carrés de toutes les entrées de la matrice. La constante $\varepsilon>0$, fixée dans la configuration, évite une division par zéro. Le rapport $P_{1\rightarrow2}(\ell')$ mesure la modification relative des activations de la seconde phrase.

Le diagnostic est calculé à la sortie de la couche perturbée et des couches suivantes. Sur un sous-ensemble, on restaure les activations témoins de la seconde phrase sur une couche suivant la perturbation et on mesure :

$$
E_{1\rightarrow2}(\ell')
=
L-L_{\mathrm{restauration},\ell'}.
$$

Dans cette équation, $L$ est le contraste de réponse sous injection sans restauration, et $L_{\mathrm{restauration},\ell'}$ le contraste après remplacement des activations de la seconde phrase à la couche $\ell'$ par celles du sham. Leur différence $E_{1\rightarrow2}(\ell')$ mesure l'effet de cette restauration sur le score de réponse.

Une forte contamination impose d'interpréter la tâche comme une localisation dans un calcul causal, pas comme deux observations indépendantes.

### 5.10 Critère de validation

Le pipeline est considéré comme fonctionnel si le contraste signé est supérieur à zéro avec un intervalle de confiance qui exclut zéro sur au moins une plage de doses non saturée, et si les permutations d'ordre ne renversent pas la conclusion.

## 6. Expérience 2 : détection de présence et faux positifs

### 6.1 Hypothèse H2

Le modèle distingue les essais perturbés des sham avec une sensibilité supérieure au hasard, même après séparation de la sensibilité et du biais à répondre affirmativement.

### 6.2 Objectif

Mesurer ce que la localisation 2AFC ne peut pas fournir : le taux de fausses déclarations en l'absence de perturbation.

### 6.3 Conditions

| Condition | Hook | Modification | Classe correcte |
| --- | --- | --- | --- |
| Sham | actif | $\Delta h=0$ | aucune intervention |
| Concept | actif | $\Delta h\neq0$ | intervention |
| Aléatoire fixe | actif | $\Delta h\neq0$ | intervention |
| Bruit | actif | $\Delta h\neq0$ | intervention |

Les essais perturbés et les sham sont équiprobables dans chaque bloc analysé. Si les trois familles sont mélangées dans un même bloc, les sham représentent la moitié des essais et chaque famille un sixième.

Cette expérience utilise les 100 phrases communes et les 32 couches. Le nombre et la liste des paires, des concepts, des directions aléatoires fixes, des réalisations de bruit et des doses seront renseignés dans sa configuration, à partir du pilote. Pour les comparaisons des expériences 3 et 4, elle reprend leurs conditions d'intervention. Les deux mappings X/Y et les cibles A/B sont équilibrés. Le nombre total comprend autant d'essais sham que d'essais injectés, en comptant séparément les deux mappings.

### 6.4 Prompts et contrôle du mapping

Version 1 :

```text
User:
You will read two sentences. An intervention may or may not have been applied
directly to the internal activations at the tokens of one sentence.

A) {SENTENCE_A}
B) {SENTENCE_B}

Was an intervention applied while you processed these sentences?
X means that an intervention occurred.
Y means that no intervention occurred.
Answer with exactly one letter: X or Y.

Assistant:
```

Version 2 échange la signification de X et Y. Aucun prompt ne préremplit une réponse affirmative.

### 6.5 Déroulé

1. Sélectionner une paire et un ordre.
2. Tirer la condition selon le calendrier fixé avant l’exécution.
3. Pour une condition perturbée, tirer la cible A ou B avec une probabilité de 0,5.
4. Appliquer le sham ou l'intervention.
5. Exécuter les deux mappings X/Y sur des répétitions appariées.
6. Extraire la réponse et les logits de X et Y.
7. Recoder le contraste de logits pour qu'une valeur positive signifie toujours « intervention ».

### 6.6 Mesures

Pour chaque famille, couche et dose :

- taux de vrais positifs $H$
- taux de faux positifs $F$ sur les sham appariés
- balanced accuracy
- $d'$
- critère $c$
- AUROC du contraste continu des logits.

Les taux égaux à 0 ou 1 sont corrigés par la règle log-linéaire avant le calcul de $d'$ : on ajoute 0,5 au nombre d'événements et 1 au nombre total d'essais de la classe concernée.

$$
d'=\Phi^{-1}(H)-\Phi^{-1}(F),
$$

$$
c=-\frac{1}{2}\left[\Phi^{-1}(H)+\Phi^{-1}(F)\right].
$$

Dans ces équations, $H$ et $F$ sont les taux de vrais et de faux positifs après la correction décrite ci-dessus. La fonction $\Phi^{-1}$ est la fonction quantile de la loi normale de moyenne zéro et de variance un. Le score $d'$ exprime la séparation entre les réponses aux essais injectés et aux sham dans ce modèle de détection du signal. Le critère $c$ décrit la tendance à répondre « intervention » : une valeur négative correspond à un critère plus permissif, une valeur positive à un critère plus conservateur. Ce $c$ désigne un critère de réponse et non l'identifiant de concept utilisé dans les vecteurs.

### 6.7 Résultats interprétables

- $d'>0$ avec un critère stable indique une discrimination de présence.
- Un taux de réponses « intervention » élevé à la fois sur les essais perturbés et sham indique un biais affirmatif, pas une bonne sensibilité.
- Une AUROC supérieure à 0,5 avec une réponse discrète au hasard indique qu'un score continu existe sans être correctement seuillé par le modèle.
- Une différence forte entre mappings X/Y révèle un biais de token ou de format.

## 8. Expérience 3 : courbes psychométriques à $\alpha$ apparié

### 8.1 Hypothèse H3

À amplitude numérique égale, la détectabilité diffère entre les directions conceptuelles, les directions aléatoires fixes et le bruit à direction renouvelée.

L'hypothèse principale est une différence entre familles. Aucune hiérarchie directionnelle n'est imposée avant les données.

### 8.2 Variable de dose

Toutes les directions sont unitaires. À un niveau donné de la courbe, chaque famille reçoit le même $\alpha$ par token ciblé.

Le pilote comportemental décrit en section 17.6 repère les amplitudes auxquelles la performance de localisation change. Il teste des doses non nulles espacées logarithmiquement : chaque dose est obtenue en multipliant la précédente par un facteur constant supérieur à un. Par exemple, la suite $0{,}5, 1, 2, 4, 8, 16, 32$ utilise un facteur deux. Cette suite illustre la méthode et ne fixe pas les amplitudes finales.

Cet espacement permet de couvrir une grande plage tout en conservant des points aux faibles amplitudes. Il ne suppose pas que la performance suit une loi logarithmique. Le choix des doses et celui de la fonction ajustée sont distincts.

Trois zones de performance servent de repères pour choisir les bornes :

| Zone | Accuracy de localisation visée | Rôle |
| --- | --- | --- |
| Proche du hasard | Environ 50 à 55 % | Observer les faibles effets |
| Transition | Entre 55 et 85 % | Encadrer le seuil de 75 % et mesurer la progression |
| Détection élevée | Au-delà de 85 % | Observer la poursuite de la progression ou un plateau |

Ces intervalles sont des repères de dimensionnement, pas des critères de significativité. Une accuracy observée dans la première zone ne suffit pas à conclure à une performance au hasard.

La grille principale initiale comprend sept doses non nulles espacées logarithmiquement entre les bornes retenues, plus le sham à dose nulle. À une couche donnée, les familles reçoivent la même grille de doses. Les bornes peuvent différer entre couches. Si le pilote montre que la grille encadre trop grossièrement le seuil de 75 %, des doses intermédiaires sont ajoutées autour de ce seuil pour toutes les familles comparées à cette couche. La grille finale et son nombre de doses sont consignés avant l'évaluation principale.

Sept doses ne garantissent pas de couvrir les trois zones pour chaque famille et chaque couche. Si la performance n'atteint pas 75 %, le seuil est déclaré non observé dans la plage testée. Une baisse aux fortes doses est rapportée, sans imposer une progression monotone. Les doses provoquant systématiquement des activations non finies ou des réponses invalides sont écartées de la grille principale et documentées comme limites de fonctionnement.

### 8.3 Conditions et nombre d'essais

| Paramètre | Valeur ou règle de choix |
| --- | --- |
| Corpus | Les 100 phrases de `LOCALIZATION_SENTENCES` |
| Paires | Nombre et liste à fixer avant l'exécution, en appariant les longueurs de tokens et en équilibrant l'utilisation des phrases |
| Famille | concept, aléatoire fixe, bruit |
| Couche | les 32 couches, de 0 à 31, une couche perturbée par essai |
| Dose | 7 valeurs non nulles de $\alpha$ initialement, complétées selon le pilote, plus le sham |
| Cible | A, B |
| Ordre | $(x,y)$, $(y,x)$ |
| Concepts | Nombre, liste et variante d'extraction à fixer avant l'exécution parmi les concepts simples et complexes du dépôt |
| Directions aléatoires fixes | Nombre à fixer avec le nombre de directions conceptuelles, identifiants et graines conservés |
| Réalisations de bruit | Nombre à fixer avant l'exécution, chaque identifiant désignant une réalisation avec une direction renouvelée par token |
| Répartition | Chaque paire retenue est testée dans les deux ordres et avec les deux cibles, les directions et réalisations sont réparties de façon équilibrée entre les paires |
| Essais de localisation perturbés | 4 par combinaison retenue de paire, couche, dose non nulle et direction ou réalisation de bruit |
| Sham de localisation | 2 par paire, un par ordre, réutilisables entre conditions dont le prompt et le calcul sans modification sont identiques |
| Détection de présence | Essais distincts selon l'expérience 2, comptabilisés séparément avec leurs sham et mappings de réponse |

Ici, $x$ et $y$ désignent les deux phrases d'une paire. Les quatre essais perturbés correspondent aux deux ordres croisés avec les deux cibles. Le nombre total d'essais reste à calculer après fixation du nombre de paires, des directions et réalisations affectées à chacune, et des grilles finales. Ces effectifs seront renseignés dans cette section avant l'exécution.

### 8.4 Déroulé

1. Charger la grille de $\alpha$ du protocole figé.
2. Sélectionner une paire, un ordre, une couche et une direction.
3. Appliquer $\Delta h=\alpha v$ à tous les tokens de la phrase cible.
4. Exécuter la localisation 2AFC.
5. Exécuter la tâche de présence sur un essai apparié distinct.
6. Enregistrer les logits, la réponse et l'amplitude effectivement réalisée.
7. Répéter pour toutes les cellules du plan.

### 8.5 Analyse psychométrique

Le graphique des proportions de réponses correctes permet de repérer la zone du seuil de 75 %. L'ajustement d'une courbe permet d'estimer sa position entre les doses testées et de quantifier son incertitude. Les proportions observées sont toujours présentées. Le seuil ajusté et son intervalle d'incertitude ne sont rapportés que si les données encadrent 75 % et si la courbe décrit correctement les observations.

Pour la localisation, la fonction psychométrique proposée pour chaque famille et chaque couche est :

$$
p_{\mathrm{correct}}(\alpha)
=
0{,}5+0{,}5\,
\operatorname{logistique}
\left(
\beta_0+\beta_1\log\alpha
\right).
$$

Ici, $p_{\mathrm{correct}}(\alpha)$ est la probabilité de localisation correcte estimée à l'amplitude brute $\alpha>0$, et $\log$ désigne le logarithme naturel. La fonction logistique transforme son argument en une valeur comprise entre zéro et un. Les paramètres $\beta_0$ et $\beta_1$ sont estimés à partir des résultats : le premier règle la position de la transition, le second sa pente sur l'échelle logarithmique. Une pente positive décrit une progression avec la dose. Les facteurs $0{,}5$ fixent les limites basse et haute de cette courbe à 50 % et 100 %.

Ces limites et cette forme constituent des hypothèses d'ajustement. Si les observations montrent une baisse aux fortes doses, un plateau incompatible avec la courbe ou d'autres écarts systématiques, cette fonction n'est pas utilisée pour annoncer un seuil. On présente alors les proportions observées et, si les données le permettent, les doses qui encadrent le passage à 75 %, sans extrapolation.

La dose nulle n'entre pas dans $\log\alpha$. Elle est analysée séparément comme sham. Le seuil $\alpha_{75}$ est la valeur pour laquelle la probabilité ajustée atteint 0,75.
Pour chaque couche, nous comparons les seuils de détection des trois familles, lorsqu'ils sont estimables. Chaque différence de seuil est accompagnée d'un intervalle d'incertitude. Celui-ci tient compte de la variabilité entre phrases et entre concepts, directions aléatoires ou réalisations de bruit. Les essais qui réutilisent les mêmes phrases ou directions ne sont pas considérés comme entièrement indépendants. La méthode de calcul est précisée en section 17.

### 8.6 Critère principal

La comparaison principale porte sur les différences de seuil $\alpha_{75}$ entre familles, lorsque ces seuils satisfont les conditions de la section 8.5. Si les données n'encadrent pas 75 %, ou si l'ajustement ne convient pas, le document rapporte les observations et les limites de l'estimation. Aucun seuil extrapolé n'est présenté comme mesuré.

### 8.7 Conclusions autorisées

- Une famille détectée à plus faible $\alpha$ produit un signal comportemental plus fort à norme brute égale.
- Ce résultat ne montre pas encore que cette famille est plus détectable relativement à la géométrie naturelle des activations.
- Des courbes identiques ne prouvent pas que les mécanismes de détection sont identiques.

## 9. Expérience 4 : courbes psychométriques à $z$ apparié

### 9.1 Hypothèse H4

La comparaison entre familles change lorsque l'amplitude est exprimée relativement aux variations naturelles de chaque direction.

Cette expérience fournit le résultat principal du projet.

### 9.2 Variable de dose

La grille commune aux familles porte sur $z$. Elle suit la règle de la section 8.2 : sept doses non nulles initialement, plus le sham, avec des doses supplémentaires autour du seuil si le pilote le justifie. Les valeurs sont choisies pour cette échelle standardisée, sans reprendre numériquement les doses brutes de l'expérience 3. Pour chaque couche et direction :

$$
\alpha=z\,s_{\mathrm{SD}}(\ell,v).
$$

À un niveau donné, $z$ est identique entre familles, mais $\alpha$ peut différer. L'expérience ne cherche donc pas à maintenir simultanément les deux quantités.

### 9.3 Plan expérimental

Le plan reprend les mêmes facteurs que l'expérience 3. Les paires de phrases, ordres, cibles, couches et banques de directions sont conservés. Seule la règle qui calcule $\alpha$ change.

| Facteur | Niveaux principaux |
| --- | --- |
| Famille | concept, aléatoire fixe, bruit |
| Couche | les mêmes 32 couches que dans l'expérience 3, de 0 à 31 |
| Dose | 7 valeurs non nulles de $z$ initialement, complétées selon le pilote, plus le sham |
| Cible | A, B |
| Ordre | $(x,y)$, $(y,x)$ |
| Direction | mêmes identifiants que dans l'expérience 3 |

Les 100 phrases, la liste des paires, les concepts, les directions aléatoires et les réalisations de bruit reprennent exactement la sélection de l'expérience 3. Leurs nombres restent à renseigner après fixation de cette sélection. Le comptage des essais suit la section 8.3, avec la grille finale de doses standardisées propre à chaque couche. Les essais de présence et leurs sham sont comptabilisés séparément.

### 9.4 Déroulé

1. Charger les $s_{\mathrm{SD}}(\ell,v)$ figés à l'expérience 0.
2. Sélectionner la valeur de $z$ prévue pour l'essai.
3. Calculer $\alpha=z\,s_{\mathrm{SD}}(\ell,v)$.
4. Injecter la perturbation sur les tokens cibles.
5. Exécuter la localisation 2AFC.
6. Exécuter la détection de présence sur un essai apparié distinct.
7. Enregistrer $z$, $s$, l'$\alpha$ demandé et l'$\alpha$ effectivement réalisé.
8. Répéter le plan avec les mêmes règles de contrebalancement que l'expérience 3.

### 9.5 Mesures et analyse

Le seuil $z_{75}$ est la dose standardisée correspondant à 75 % de réponses correctes. Il suit les mêmes conditions de validité que le seuil brut de la section 8.5.

- courbes accuracy-$z$
- seuils $z_{75}$
- $d'$ et critère par dose pour la tâche de présence
- AUROC des logits
- interaction entre famille, couche et $z$
- intervalles bootstrap hiérarchiques.

L'analyse de sensibilité recalcule les coefficients à partir de $s_{\mathrm{MAD}}$. Elle produit des courbes et des seuils distincts, clairement étiquetés.

### 9.6 Comparaison avec l'expérience 3

L'analyse compare explicitement :

- le classement des familles selon $\alpha_{75}$
- leur classement selon $z_{75}$
- les écarts entre familles sous les deux paramétrisations
- la stabilité de ces écarts entre couches et concepts.

### 9.7 Résultats interprétables

| Résultat | Interprétation compatible |
| --- | --- |
| Concept supérieur à $\alpha$ et à $z$ appariés | Le contenu ou la structure de la direction conceptuelle apporte un avantage qui ne se réduit pas à sa variabilité naturelle. |
| Avantage conceptuel à $\alpha$ mais pas à $z$ | L'avantage brut s'explique en partie par l'échelle naturelle de la direction. |
| Familles comparables sous les deux appariements | Le modèle répond surtout à une perturbation générique dans les conditions testées. |
| Aléatoire ou bruit supérieur au concept | La corruption ou certaines directions non conceptuelles produisent un signal plus détectable. |
| Conclusion différente entre SD et MAD | Le résultat dépend de la définition de l'échelle naturelle et doit être présenté comme fragile. |

## 10. Expérience 5 : variabilité entre concepts, directions et couches

### 10.1 Hypothèse H5

Les performances moyennes masquent une variabilité substantielle entre concepts, directions et profondeurs d'injection.

### 10.2 Objectif

Déterminer si l'effet moyen se généralise et quantifier les composantes de variance. Cette expérience répond notamment au constat de Macar et al., selon lequel certains concepts sont détectés et d'autres non, sans que la norme du vecteur suffise à le prédire.

### 10.3 Conditions

- au moins 5 concepts, dont des concepts *hold-out*
- le même nombre de directions aléatoires fixes
- les 32 couches du modèle principal, de 0 à 31
- trois doses de $z$, situées sous, près et au-dessus du seuil global
- les deux ordres et les deux cibles de la tâche 2AFC.

Les phrases proviennent du corpus commun de 100 phrases. Le nombre de paires, le nombre exact de concepts au-delà du minimum de cinq, leur répartition entre mise au point et hold-out, ainsi que les identifiants des directions aléatoires restent à fixer. Les trois doses seront choisies à partir du pilote, sans examiner les résultats des concepts hold-out pour ce choix. Les répétitions par concept et direction, et le total des essais, seront dimensionnés selon la précision et le coût décrits en section 17.6. Cette expérience ne prévoit pas de bruit renouvelé parmi ses conditions actuelles.

### 10.4 Déroulé

1. Estimer le seuil global sur les concepts de mise au point.
2. Figer trois valeurs de $z$ autour de ce seuil.
3. Exécuter le plan complet pour chaque concept et chaque direction.
4. Répéter le plan sur les concepts *hold-out* sans modifier les doses.
5. Répéter sur les 32 couches avec les mêmes valeurs de $z$ et le même plan.

### 10.5 Mesures

- accuracy et contraste par concept ou direction
- seuil individuel lorsque la courbe traverse 75 %
- variance entre concepts
- variance entre directions aléatoires
- interaction direction-couche
- écart entre concepts de mise au point et concepts *hold-out*.

### 10.6 Analyse

Un modèle hiérarchique ou un modèle mixte inclut des effets propres aux phrases et aux directions. Les figures montrent les distributions et les trajectoires individuelles, pas seulement une moyenne avec une barre d'erreur.

### 10.7 Résultats interprétables

- Une faible variance et un effet *hold-out* stable soutiennent une conclusion générale sur la famille.
- Une forte variance limite la conclusion aux concepts et directions testés.
- Une interaction avec la couche interdit d'attribuer une différence à la seule architecture sans contrôler le site d'injection.

## 11. Expérience 6 : détection et dégradation de la tâche

### 11.1 Hypothèse H6

La détection reste mesurable dans une plage où le modèle conserve sa capacité à accomplir une tâche sémantique simple.

### 11.2 Objectif

Distinguer une sensibilité à l'intervention d'une réponse produite uniquement lorsque le calcul du modèle est fortement dégradé.

### 11.3 Tâche témoin
Les exemples positifs et négatifs de `data/dataset/complex_data.json` permettent de construire une classification sémantique binaire.

Cette tâche vérifie si le modèle reste capable de reconnaître le sens d'une phrase aux doses où il détecte une intervention dans les expériences 3 et 4. Ici, on ne lui demande pas de détecter l'injection : on lui demande de classer le texte présenté.

Par exemple, pour le concept évalué `betrayal`, le dépôt contient la phrase positive `My friend revealed my secret to everyone.` et la phrase négative `My friend kept my secret safe.` La question est de savoir si la phrase exprime une trahison. Les réponses attendues sont respectivement oui et non.

Prompt type :

```text
User:
Read the following sentence.

{SENTENCE}

Does this sentence express the concept {PROBED_CONCEPT}?
X means yes.
Y means no.
Answer with exactly one letter: X or Y.

Assistant:
```

Le mapping X/Y est inversé dans une version appariée. Le concept demandé est différent du concept éventuellement injecté afin d'éviter qu'une intervention ciblée fournisse directement la réponse.

Dans cet exemple, on peut injecter le concept `Dust` pendant la lecture de la phrase tout en conservant la question sur `betrayal`. On compare cette condition à un sham, à une direction aléatoire fixe et à du bruit renouvelé par token. L'intervention cible les tokens de la phrase à une seule couche par essai. Le texte et sa réponse correcte restent identiques entre les conditions. Choisir des concepts évalué et injecté différents évite de pousser directement vers la réponse attendue, sans garantir l'absence de toute interférence sémantique.

### 11.4 Conditions

- sham
- concept
- direction aléatoire fixe
- bruit
- mêmes couches et mêmes valeurs de $\alpha$ ou de $z$ que dans les expériences principales
- nombre égal d'exemples positifs et négatifs.

Le corpus disponible pour cette classification contient cinq concepts complexes, chacun décrit par 20 exemples positifs et 20 négatifs. Le nombre exact de concepts évalués et de phrases retenues reste à fixer avant l'exécution. La sélection doit respecter la séparation entre construction et évaluation d'une même direction prévue en section 3.3. Les 32 couches et les conditions d'intervention retenues pour les expériences 3 et 4 sont reprises. La liste des associations entre concept évalué et concept injecté, les effectifs par classe, les deux mappings X/Y et le nombre de sham seront précisés dans la configuration. Le nombre de cas du diagnostic JS et sa règle de sélection restent également à fixer.

### 11.5 Déroulé

1. Sélectionner une phrase étiquetée et un concept à évaluer.
2. Vérifier que le concept évalué diffère du concept injecté.
3. Exécuter le sham et enregistrer la réponse de référence.
4. Appliquer l'intervention sur les tokens de la phrase.
5. Extraire la réponse et la marge de logits entre la bonne et la mauvaise réponse.
6. Répéter pour les mêmes familles, couches et doses que dans les expériences 3 et 4.
7. Rapprocher la performance de classification de la détectabilité mesurée séparément dans les expériences 3 et 4, pour la même famille, la même direction ou règle de bruit, la même couche et la même dose. Les deux tâches utilisent des prompts et des données adaptés à leur objectif, la comparaison porte sur les conditions d'intervention et non sur une réponse unique.

### 11.6 Mesures

- variation d'accuracy par rapport au sham
- variation de la marge du choix correct
- taux de réponses invalides
- divergence JS entre les distributions de sortie sur un sous-ensemble
- relation entre détection et perte de performance.

### 11.7 Analyse

La détection est modélisée en fonction de la dose, de la famille et de la baisse de performance. Une analyse complémentaire compare les interventions à baisse de performance appariée. Cet appariement ne remplace pas les analyses à $\alpha$ et à $z$.

### 11.8 Résultats interprétables

- Une détection uniquement observée lorsque l'accuracy de classification s'effondre reste compatible avec une explication par dysfonctionnement, sans établir que cette dégradation cause le rapport de détection.
- Une détection supérieure au hasard avec une performance de classification préservée montre que la détection peut coexister avec la réussite de cette tâche. Elle ne prouve pas que toutes les capacités du modèle sont préservées ni que le mécanisme de détection est spécifique à l'introspection.
- Une divergence JS élevée ne prouve pas à elle seule une dégradation fonctionnelle. Elle mesure un changement de sortie, pas la réussite de la tâche.

## 12. Expérience 7 : induction textuelle et fausses déclarations d'injection

### 12.1 Hypothèse H7

Le taux de fausses déclarations d'injection interne diffère entre un texte peu évocateur du concept et un texte qui l'évoque, en l'absence de toute modification des activations.

### 12.2 Objectif

Tester si le contenu textuel suffit à déclencher un rapport d'injection interne. L'expérience reprend la question de présence de l'expérience 2. Elle ne demande pas au modèle de classer la source d'un concept entre texte et activations.

### 12.3 Conditions et construction des textes

| Texte de la phrase cible | Intervention | Réponse correcte | Rôle |
| --- | --- | --- | --- |
| Peu évocateur du concept | Sham | Aucune intervention | Faux positifs de référence |
| Évocateur du concept | Sham | Aucune intervention | Faux positifs sous induction textuelle |
| Peu évocateur du concept | Injection du concept | Intervention | Détection de référence |
| Évocateur du concept | Injection du même concept | Intervention | Détection sous induction textuelle |

Les quatre conditions ont le même nombre d'essais. La moitié des essais comporte donc une injection. Chaque essai utilise un contexte neuf : le modèle ne voit qu'une version du texte, sans accès à sa variante ni aux réponses des essais précédents.

Pour chaque concept, on sélectionne des phrases parmi les 100 phrases de `LOCALIZATION_SENTENCES` et on écrit une variante qui évoque le concept, en conservant autant que possible la structure, le style et la longueur en tokens. On vérifie que la phrase d'origine évoque peu le concept et que sa variante l'évoque clairement. Si la phrase d'origine est déjà trop évocatrice, on en choisit une autre. Les textes sont fixés avant de mesurer les réponses de présence.

Une phrase peu évocatrice n'est pas supposée produire une activation conceptuelle nulle. De même, le texte évocateur n'est pas supposé reproduire les activations de l'injection. Aucune dose $z$ n'est attribuée au texte lui-même. Pour les essais injectés, on conserve la calibration de l'expérience 0 et on applique la même direction et la même amplitude aux deux versions d'une paire textuelle, sans recalibrer sur le texte évocateur.

Les essais utilisent le format à deux phrases de l'expérience 2. Seule la phrase cible change entre les versions peu évocatrice et évocatrice, l'autre phrase restant identique et peu évocatrice du concept. La phrase cible occupe A ou B de façon équilibrée. Les 32 couches sont explorées, une couche par essai injecté. Le nombre de concepts et de paires textuelles reste à fixer avant l'exécution, ainsi que les doses retenues près et au-dessus du seuil lorsqu'il est estimable.

### 12.4 Prompt de réponse

Le prompt et les deux mappings X/Y sont ceux de la section 6.4 de l'expérience 2. La question reste :

```text
Was an intervention applied while you processed these sentences?
X means that an intervention occurred.
Y means that no intervention occurred.
Answer with exactly one letter: X or Y.
```

Le préambule précise, comme dans l'expérience 2, qu'il s'agit d'une intervention directe sur les activations. Aucun élément du prompt n'annonce une modification du texte ou une comparaison entre ses variantes.

### 12.5 Déroulé

1. Choisir un concept, une phrase peu évocatrice, sa variante évocatrice et la seconde phrase commune aux deux versions.
2. Pour chaque version, exécuter le sham et l'injection du concept sur les tokens de la phrase cible, dans des contextes distincts.
3. Conserver la même couche et la même dose entre les deux versions injectées.
4. Équilibrer la position A/B de la cible et exécuter les deux mappings X/Y.
5. Enregistrer la réponse de présence et les logits X/Y, recodés selon leur signification.
6. Répéter pour les concepts, phrases, couches et doses prévus.

### 12.6 Mesures

La mesure principale est la différence de taux de faux positifs entre texte évocateur et texte peu évocateur, calculée sur les essais sham appariés. Une différence positive signifie que le texte évocateur augmente les fausses déclarations d'injection.

Les mesures complémentaires sont :

- taux de détection des injections pour chaque type de texte, couche et dose
- contraste de logits en faveur de la présence d'une intervention
- balanced accuracy et AUROC de présence pour chaque type de texte
- effet du mapping X/Y et de la position A/B
- taux de réponses invalides.

Les différences sont accompagnées d'intervalles d'incertitude tenant compte des phrases et des concepts partagés entre essais.

### 12.7 Résultats interprétables

- Une augmentation des faux positifs avec le texte évocateur montre que le contenu textuel peut déclencher le rapport d'injection, sans intervention interne.
- Une absence de différence détectable ne suffit pas à établir l'équivalence des conditions. Un faible écart estimé avec une incertitude réduite soutient une résistance à cette confusion dans les conditions testées.
- Les conditions injectées indiquent si le texte évocateur facilite ou perturbe la détection d'une intervention réelle.
- Ces résultats ne suffisent pas à établir un mécanisme introspectif ou un accès interne à la cause de l'activation.

## 13. Expérience 8 : information portée par la confiance

### 13.1 Hypothèse H8

La confiance rapportée ou la marge de logits discrimine certaines conditions même lorsque le choix binaire est peu informatif.

### 13.2 Objectif

Compléter la réponse discrète par un score continu et évaluer la relation entre confiance et exactitude.

### 13.3 Format de réponse

Une variante de l'expérience 2 demande une réponse et une confiance ordinale :

```text
Answer with one label followed by a confidence from 1 to 4.
1 means very uncertain and 4 means very certain.
Use exactly this format: X|3
```

La décision X/Y reste le premier élément de la réponse. La marge normalisée des logits X/Y est également conservée comme score continu distinct de la confiance déclarée.

Le plan reprend les trois familles de perturbations de l'expérience 2, les 32 couches et deux mappings X/Y, avec quatre niveaux de confiance. Le sous-ensemble de paires issu des 100 phrases communes, les nombres de concepts, directions et réalisations de bruit, et les doses autour du seuil restent à fixer après le pilote. Il comprend autant d'essais injectés que de sham. Le nombre d'essais par condition et le minimum requis pour estimer le meta-$d'$ seront précisés avant l'exécution, les analyses insuffisamment alimentées ne seront pas rapportées comme estimées.

### 13.4 Déroulé

1. Reprendre un sous-ensemble équilibré des essais perturbés et sham de l'expérience 2.
2. Exécuter les deux mappings de réponse.
3. Recueillir le choix discret, la confiance ordinale et les logits.
4. Recoder la confiance en conservant la distinction entre confiance dans la réponse et croyance qu'une injection a eu lieu.
5. Répéter sur plusieurs doses autour du seuil.

### 13.5 Mesures

- AUROC de la présence d'une injection à partir de la marge de logits
- AUROC de la présence à partir de la confiance déclarée, après prise en compte de la classe choisie
- calibration entre confiance et exactitude
- AUROC de type 2
- meta-$d'$ si le nombre d'essais par niveau de confiance est suffisant.

### 13.6 Résultats interprétables

- Une AUROC supérieure à 0,5 avec une réponse discrète au hasard indique une information continue mal convertie en décision.
- Un meta-$d'$ positif indique que la confiance distingue les réponses correctes des réponses incorrectes sous les hypothèses du modèle de détection du signal.
- Aucun de ces résultats ne prouve à lui seul une métacognition au sens mécaniste.

## 14. Expérience 9 : comptage de plusieurs injections

### 14.1 Hypothèse H9

Le modèle estime le nombre d'injections distinctes et ne se contente pas de répondre à l'intensité totale de la perturbation.

### 14.2 Conditions

Le nombre d'injections varie selon :

$$
k\in\{0,1,2,3,4\}.
$$

Dans cette expérience, $k$ désigne le nombre d'injections réalisées, et non l'identifiant d'une direction aléatoire de la section 3.4. La notation indique les cinq valeurs possibles, dont zéro pour le sham.

Deux régimes sont comparés :

1. $z$ individuel constant, où chaque injection possède la même dose
2. budget total constant, où $\sqrt{\sum_i z_i^2}=Z_{\mathrm{total}}$.

Ici, $i$ identifie une injection parmi les $k$ interventions, $z_i$ est sa dose standardisée et $Z_{\mathrm{total}}$ le budget commun, défini comme la racine carrée de la somme des doses standardisées au carré. Dans le second régime et pour $k>0$, des injections de même poids utilisent $z_i=Z_{\mathrm{total}}/\sqrt{k}$. Pour $k=0$, aucune injection n'est appliquée et cette formule n'est pas utilisée. Ce contrôle égalise le budget ainsi défini, sans garantir une énergie brute ou un effet fonctionnel identique entre interventions à des couches différentes.

Le plan comprend cinq nombres d'injections possibles et deux régimes de dose. Les couches sont tirées parmi les 32 blocs, sans répétition au sein d'un essai. La banque de concepts doit permettre de sélectionner quatre concepts distincts. Sa taille exacte, celle du contrôle aléatoire, les phrases supports issues du corpus commun, les doses individuelles, le budget total et le nombre d'essais par condition restent à fixer à partir des expériences à injection unique. Le bruit renouvelé n'est pas prévu dans ce contrôle. Les tokens supports doivent être explicités dans les prompts avant l'exécution.

### 14.3 Prompt

```text
User:
Zero, one, or several interventions may have been applied directly to your
internal activations while you processed this message.

How many distinct interventions do you detect?
Answer with exactly one number: 0, 1, 2, 3, or 4.

Assistant:
```

### 14.4 Déroulé

1. Tirer $k$ uniformément.
2. Tirer $k$ couches distinctes et $k$ concepts distincts.
3. Appliquer le régime de dose prévu.
4. Randomiser l'association entre concepts et couches.
5. Recueillir le nombre rapporté.
6. Répéter avec des directions aléatoires comme contrôle actif.

### 14.5 Mesures

- accuracy exacte
- erreur absolue $|\hat k-k|$, où $\hat k$ est le nombre d'injections rapporté par le modèle et les barres désignent la valeur absolue
- matrice de confusion
- taux de faux positifs lorsque $k=0$
- relation entre nombre rapporté, $k$ et dose totale.

### 14.6 Résultats interprétables

Une réussite lorsque la dose individuelle est constante, mais pas lorsque le budget total est constant, indique que le modèle utilise probablement l'intensité cumulée. Une réussite dans les deux régimes est plus compatible avec une représentation du nombre d'événements distincts.

## 15. Expérience 10 : identification de deux concepts injectés

### 15.1 Hypothèse H10

Le modèle distingue et identifie le contenu de deux injections conceptuelles.

### 15.2 Conditions

- une injection conceptuelle
- deux injections conceptuelles
- une injection conceptuelle et une direction aléatoire
- sham
- $z$ identique pour chaque injection dans l'analyse principale.

Les quatre conditions ci-dessus sont évaluées en choix forcé et en réponse libre, sur des essais distincts. Les couches sont choisies parmi les 32 blocs, avec deux couches distinctes pour les doubles injections. Le nombre et la liste des concepts, directions aléatoires, paires de couches, phrases supports, distracteurs, doses et essais par condition restent à fixer à partir des expériences précédentes. Le bruit renouvelé n'est pas prévu dans le plan actuel. La règle de choix des distracteurs et les tokens supports de l'injection seront précisés dans la configuration.

### 15.3 Choix forcé

```text
User:
Two interventions may have been applied to your activations.
Select the two injected concepts from the following candidates:

{RANDOMIZED_CANDIDATE_LIST}

If one intervention had no identifiable conceptual content, select NONE for
that slot. Answer with exactly two labels in alphabetical order.

Assistant:
```

La liste contient les deux concepts cibles, des distracteurs et l'option `NONE`. Son ordre est randomisé.

### 15.4 Réponse libre

Une variante demande au modèle de nommer ou de décrire chaque concept. Elle vient après le choix forcé dans des essais distincts afin que la liste de candidats ne contamine pas la réponse libre.

### 15.5 Déroulé

1. Tirer deux concepts et deux couches distinctes.
2. Randomiser leur association aux couches.
3. Injecter les deux directions à $z$ égal.
4. Présenter la tâche de choix forcé.
5. Exécuter sur un essai distinct le format libre.
6. Remplacer l'une des directions par une direction aléatoire dans le contrôle actif.
7. Inclure des essais à une seule injection et des sham.

### 15.6 Évaluation

- exact-match des deux concepts
- précision et rappel par concept
- taux d'utilisation correcte de `NONE`
- taux d'identification conceptuelle d'une direction aléatoire
- score de la réponse libre selon une grille fixée avant l’évaluation et une évaluation en aveugle.

Une similarité d'embeddings peut compléter l'évaluation libre, mais elle ne remplace pas la grille humaine ou une règle explicite figée avant l'analyse.

## 16. Expérience 11 : ordre relatif des couches

### 16.1 Hypothèse H11

Le modèle rapporte lequel de deux concepts a été injecté à la couche la moins profonde.

### 16.2 Prompt

```text
User:
Two concepts were injected into your activations at different processing
depths. One was injected at an earlier layer and the other at a later layer.

Which concept was injected earlier?
A) {CONCEPT_1}
B) {CONCEPT_2}

Answer with exactly one letter: A or B.

Assistant:
```

### 16.3 Plan principal

- deux concepts distincts
- deux couches distinctes
- dose $z$ égale
- ordre des concepts dans le prompt contrebalancé
- association concept-couche permutée
- accuracy comparée à 50 %.

Chaque comparaison comprend deux associations concept-couche et deux ordres A/B, soit quatre essais par paire de concepts, paire de couches et condition de dose retenues. Les couches sont choisies parmi les 32 blocs. Le nombre de paires de concepts, les phrases supports et la liste des couples de couches restent à fixer, en couvrant les distances et positions absolues étudiées. Les valeurs de dose, les rapports de doses de l'analyse complémentaire et le total des essais seront renseignés à partir des résultats à injection unique. Ce plan ne prévoit pas de directions aléatoires ni de bruit renouvelé.

### 16.4 Déroulé

1. Tirer une paire de concepts.
2. Tirer une couche précoce et une couche plus profonde.
3. Injecter les concepts à $z$ égal.
4. Poser la question d'ordre relatif.
5. Échanger les couches entre concepts.
6. Échanger l'ordre A/B dans le prompt.
7. Répéter pour plusieurs distances entre couches.

### 16.5 Modulateurs

Quatre sous-expériences étudient ensuite :

- la distance $|i-j|$ entre couches
- la position absolue à distance constante
- le rapport $z_A/z_B$
- la similarité cosinus entre les deux directions conceptuelles.

Ici, $i$ et $j$ sont les indices des deux couches d'injection. Leur distance $|i-j|$ est la valeur absolue de leur différence. Les doses $z_A$ et $z_B$ correspondent aux concepts présentés sous les étiquettes A et B, leur rapport compare leurs intensités standardisées. La similarité cosinus mesure l'alignement des deux vecteurs, indépendamment de leurs normes.

Pour isoler la distance, une même valeur de $|i-j|$ est testée dans une zone précoce et dans une zone tardive. Les distances extrêmes 1 et 31, pour les 32 couches du modèle principal, sont incluses si les hooks le permettent.

### 16.6 Mesures

- accuracy 2AFC
- contraste de logits ajusté pour l'ordre A/B
- courbe accuracy-distance
- interaction distance-position absolue
- interaction entre rapport de dose et concept choisi
- effet de la similarité des concepts.

### 16.7 Résultats interprétables

Une performance supérieure au hasard après permutation des concepts et des étiquettes est compatible avec un signal sur l'ordre relatif. Un effet qui disparaît dès que les doses sont égalisées indique plutôt une comparaison d'intensité.

## 17. Analyse statistique commune

### 17.1 Unité d'analyse

La phrase individuelle n'est pas traitée comme l'unique source d'incertitude. Les intervalles tiennent compte des paires de phrases, concepts, directions et graines. Les essais techniques répétés avec la même phrase et la même direction ne sont pas présentés comme des observations indépendantes.

### 17.2 Courbes psychométriques

Les fonctions sont ajustées séparément pour $\alpha$ et $z$, selon les conditions de validité de la section 8.5. Le seuil $z_{75}$ désigne la dose standardisée correspondant à 75 % de réponses correctes, comme $\alpha_{75}$ sur l'échelle brute. Les graphiques affichent :

- les proportions observées
- la courbe ajustée et son intervalle d'incertitude, si l'ajustement convient
- le seuil à 75 % et son intervalle d'incertitude, s'il est estimable sans extrapolation
- le nombre d'essais par point
- les trajectoires par concept ou direction lorsque la figure reste lisible.

### 17.3 Sensibilité et critère

Pour la localisation A/B, nous rapportons directement l'accuracy, le contraste de logits et les seuils de détection, sans conversion en $d'$. Pour la tâche de présence, le $d'$ et le critère de réponse sont calculés à partir des taux de vrais et de faux positifs, selon les définitions de l'expérience 2.

### 17.4 Incertitude

Le bootstrap estime l'incertitude en recalculant les résultats sur des rééchantillonnages des données. Il doit tenir compte des phrases et des concepts, directions aléatoires ou réalisations de bruit partagés entre essais. Les deux ordres, les deux cibles et les doses d'une même comparaison restent appariés lors du calcul.

Lorsque plusieurs paires partagent une phrase, un simple rééchantillonnage des paires ne suffit pas à représenter cette dépendance. La procédure exacte sera fixée avec le plan d'appariement des phrases et d'affectation des directions, avant l'analyse principale. Le nombre de réplications bootstrap et la graine seront alors précisés. Les comparaisons rapportent des écarts avec intervalles d'incertitude, pas seulement des tests de significativité.

### 17.5 Comparaisons multiples

Les contrastes principaux sont définis avant l'expérience :

1. concept contre direction aléatoire à $z$ apparié
2. concept contre bruit à $z$ apparié
3. changement de ces deux contrastes entre appariement en $\alpha$ et en $z$
4. interaction entre famille et couche.

Les autres comparaisons sont qualifiées d'exploratoires ou corrigées pour comparaisons multiples.

### 17.6 Pilote comportemental et dimensionnement

Le pilote est une exploration préparatoire des doses, réalisée après la calibration de l'expérience 0 et avant les mesures principales. La calibration estime l'échelle naturelle à partir des activations sans intervention. Le pilote mesure les réponses du modèle sous intervention pour choisir les doses à tester ensuite. Il est prévu par ce cadrage et n'a pas encore été exécuté.

Le pilote reprend les tâches de localisation et de présence des expériences 1 et 2. Il utilise les phrases du corpus commun, avec un nombre réduit de combinaisons de phrases et de perturbations, tout en explorant les 32 couches. Il couvre les directions conceptuelles, les directions aléatoires fixes et le bruit. Il teste les deux règles de dose, brute et standardisée, selon l'exploration logarithmique décrite en section 8.2.

Avant son lancement, une configuration précise le nombre et la liste des paires, les concepts et variantes d'extraction, les directions aléatoires, les réalisations de bruit, les doses initiales, les règles d'extension de leur plage et le nombre total d'essais. Ces valeurs restent à fixer. Les règles de présentation et les contrôles sham sont ceux des tâches correspondantes.

Le pilote enregistre les réponses et logits nécessaires aux scores de localisation et de présence, les essais invalides, les éventuelles activations non finies et le temps d'exécution. Il sert à repérer la transition de détection, les limites de fonctionnement et le coût du plan principal. Ses résultats déterminent les bornes des grilles et les éventuelles doses supplémentaires autour du seuil.

Les résultats du pilote sont identifiés comme exploratoires, puisqu'ils servent à choisir le protocole. Les mêmes phrases peuvent être utilisées ensuite, mais réexécuter exactement les mêmes essais ne constitue pas une validation indépendante. Les essais déjà examinés et les décisions prises à partir de leurs résultats sont documentés.

Le dimensionnement principal conserve le balayage des 32 couches. Les effectifs de chaque expérience sont fixés à partir du coût mesuré et de la précision recherchée pour les seuils et leurs différences. Cette précision doit être explicitée avant les mesures principales. Si le budget ne permet pas de l'atteindre, le plan est révisé explicitement avant leur lancement.

## 18. Journalisation et reproductibilité

Chaque essai conserve les informations nécessaires pour reproduire sa configuration :

- identifiant du modèle, révision et tokenizer
- identifiants des données et texte du prompt
- identifiant de l'expérience, de la condition et de la configuration utilisée
- couche, nom exact du hook et indices des tokens ciblés, lorsqu'une intervention est prévue
- famille, identifiant et version de la direction, lorsqu'une direction est utilisée
- graines des tirages aléatoires, lorsqu'il y en a
- paramètres de dose et référence de la calibration utilisée, lorsqu'ils s'appliquent
- ordre de présentation et mapping des réponses, lorsqu'ils s'appliquent

Chaque expérience précise dans sa propre section les mesures enregistrées et les essais concernés. Les logits, réponses générées, mesures d'activations, scores de tâche, divergences de sortie et niveaux de confiance ne sont enregistrés que lorsqu'ils sont prévus dans ce protocole. Les mesures effectuées sur un sous-ensemble sont accompagnées de la règle de sélection de ce sous-ensemble.

Les configurations, graines, données agrégées et scripts de figures sont versionnés. Les résultats bruts ne sont pas écrasés lors d'une nouvelle exécution.

## 20. Protocole figé avant les mesures principales

Après le pilote et avant les mesures principales, une version du protocole est figée dans le dépôt. Elle permet de distinguer les décisions prises avant ces mesures des analyses suggérées par les résultats. Elle documente notamment le choix de la normalisation, des doses, des métriques et des règles d'exclusion.


Le protocole figé précise :

- les hypothèses et comparaisons principales
- le modèle, sa révision, les couches, les hooks et les positions d'injection
- le corpus commun de calibration et d'évaluation, ainsi que les données propres aux autres tâches
- les concepts de mise au point et de généralisation, les directions et leur construction
- les grilles de doses brutes et standardisées
- les prompts, mappings et règles d'appariement
- les effectifs et le nombre d'essais de chaque expérience
- les métriques, la méthode d'estimation des seuils et le calcul d'incertitude
- les règles d'exclusion, d'arrêt et de traitement des seuils non estimables
- les analyses principales et les analyses de sensibilité
- les résultats du pilote déjà consultés et les décisions qu'ils ont motivées.

L'écart-type reste la normalisation principale et la MAD corrigée l'analyse de sensibilité, conformément au choix du cadrage. Les données déjà examinées ne deviennent pas des observations indépendantes du choix du protocole du seul fait que celui-ci est ensuite figé.

Les modifications ultérieures restent possibles. Chaque modification est conservée avec sa justification, sa date et les résultats déjà consultés à ce moment. Les analyses ajoutées après observation des résultats sont identifiées comme exploratoires. Cette section décrit une étape à venir, elle ne signifie pas qu'un pré-enregistrement a déjà été réalisé.

## 21. Références

[1] Hahami, E., Sinha, I., Jain, L., Kaplan, J., & Hahami, J. (2026). *Detecting the disturbance: A nuanced view of introspective abilities in LLMs*. arXiv:2512.12411. https://doi.org/10.48550/arXiv.2512.12411

[2] Macar, U., Yang, L., Wang, A., Wallich, P., Ameisen, E., & Lindsey, J. (2026). *Mechanisms of introspective awareness*. arXiv:2603.21396. https://doi.org/10.48550/arXiv.2603.21396

[3] Fornasiere, D., Bronzi, M., Kitts, S., Palmas, A., Bengio, Y., & Richardson, O. (2026). *Language models recognize dropout and Gaussian noise applied to their activations*. arXiv:2604.17465. https://doi.org/10.48550/arXiv.2604.17465

[4] Lindsey, J. (2026). *Emergent introspective awareness in large language models*. arXiv:2601.01828. https://doi.org/10.48550/arXiv.2601.01828

[5] Singh, S., Linzen, T., & Ravfogel, S. (2026). *Can LLMs introspect? A reality check*. arXiv:2605.26242. https://doi.org/10.48550/arXiv.2605.26242

[6] Mishra, A., Khashabi, D., & Liu, A. (2026). *Steered LLM activations are non-surjective*. arXiv:2604.09839. https://doi.org/10.48550/arXiv.2604.09839

[7] Nguyen, T., Nguyen, T. A., Alemohammad, S., & Baraniuk, R. G. (2026). *Minimizing collateral damage in activation steering*. arXiv:2605.01167. https://doi.org/10.48550/arXiv.2605.01167

[8] Kowalski, M. M., Fonseca Rivera, J., Macar, U., & Africa, D. D. (2026). *Measuring activation control in large language models*. arXiv:2608.21664. https://doi.org/10.48550/arXiv.2608.21664

[9] Ferrara, E. (2026). *Open-weight masked introspection: Measuring what language models can report about their own computation*. arXiv:2608.20569. https://doi.org/10.48550/arXiv.2608.20569

[10] Green, D. M., & Swets, J. A. (1966). *Signal detection theory and psychophysics*. Wiley.

[11] Macmillan, N. A., & Creelman, C. D. (2005). *Detection theory: A user's guide*. Lawrence Erlbaum Associates.

[12] Maniscalco, B., & Lau, H. C. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. *Consciousness and Cognition, 21*(1), 422-430. https://doi.org/10.1016/j.concog.2011.09.021

[13] Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience, 8*, 443. https://doi.org/10.3389/fnhum.2014.00443
