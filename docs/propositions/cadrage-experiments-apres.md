# Cadrage expérimental révisé

Détection, localisation et spécificité des perturbations internes

> Version révisée à partir de `docs/propositions/cadrage-experiments.tex`. La version avant correspond au PDF initial. Les équations sont conservées en LaTeX dans le Markdown ; la macro locale de norme a été développée pour permettre leur affichage.

Proposition pour un hackathon de dix jours. Les choix chiffrés ci-dessous sont des paramètres de départ à figer après un pilote indépendant, pas des résultats expérimentaux.

## Question scientifique et résultats attendus

**Question principale.** À dose relative et protocole identiques, la détection et la localisation diffèrent-elles entre directions conceptuelles, directions aléatoires cohérentes et bruit renouvelé par token ? Ces différences persistent-elles après contrôle du biais de réponse, d'une manipulation textuelle et de la dégradation de la tâche ?

La contribution est un protocole commun et une comparaison interprétable. Elle ne repose ni sur la nouveauté d'un balayage couche/intensité, ni sur la découverte de la localisation de phrases, déjà étudiées. Les hypothèses suivantes sont opérationnelles et falsifiables dans le domaine testé ; leur rejet ne tranche pas la possibilité générale d'introspection.

| **Hypothèse** | **Expérience** | **Mesure décisive** | **Portée** |
| --- | --- | --- | --- |
| H1 : sensibilité | Injection contre sham, séparément par dose et couche | AUROC du score de présence ; TPR, FPR, $d'$ et critère | Discrimination sous ce protocole |
| H2 : localisation | Déplacer la même intervention entre les deux phrases | Accuracy brute A/B, effet du site, asymétrie de position | Localisation du site textuel directement ciblé |
| H3 : effet du type | Concept contre aléatoire cohérent à même dose ; bruit en comparaison secondaire | Différence d'AUROC et d'accuracy, avec IC appariés | Différence de famille, pas preuve de sémantique introspective |
| H4 : spécificité | Injection interne, manipulation textuelle, sham | Matrice de confusion ; fausses attributions internes au texte | Test d'une explication par anomalie ou suggestion |
| H5 : fonctionnement | Même texte et intervention, question de compréhension séparée | Perte de performance et variation de log-vraisemblance | Situer la détection par rapport au dysfonctionnement |

**Hiérarchie.** Critère confirmatoire principal : différence d'AUROC de présence entre concept et aléatoire cohérent, à une dose choisie sur pilote et une couche précoce préspécifiée. Le noyau comprend ces deux familles, une couche précoce, trois doses positives, les tâches de présence et de localisation, les shams, les mappings entièrement croisés et une mesure de compréhension sur un sous-plan fixé. La couche médiane, le bruit renouvelé, la quatrième dose, la normalisation en $z$, l'appariement JS, le contrôle textuel et les diagnostics causaux sont des extensions hiérarchisées. Les courbes complètes et les autres contrastes sont secondaires ; publier toutes les cellules exécutées, avec leurs intervalles. Le choix d'une seule cible principale limite la sélection du meilleur résultat après observation.

## Noyau expérimental et unité d'analyse

**Modèle.** Meta-Llama-3.1-8B-Instruct, révision et tokenizer enregistrés, mode évaluation, même précision numérique partout. Privilégier BF16 si le matériel le permet ; toute quantification devient une condition documentée. Le noyau porte sur la sortie du bloc d'indice zéro-based 3, soit le quatrième bloc ; la sortie du bloc 15, soit le seizième, constitue l'extension à une couche médiane. Cette convention évite de confondre une sortie de bloc avec l'indice d'un tableau `hidden_states`.

**Données.** Un corpus d'extraction des vecteurs, un corpus de calibration/pilote et un corpus de test disjoints par phrases. Réutiliser les ressources de Hahami pour une réplication pilote ; construire un test indépendant pour la comparaison principale. Point de départ : 40 paires de phrases de test, 6 concepts choisis avant les résultats, et 6 directions aléatoires fixes. Répartir concepts concrets et abstraits ; un petit nombre de concepts limite explicitement la généralisation. Ne pas sélectionner les concepts pour leur détectabilité. Les vecteurs existants sont réutilisables seulement si modèle, checkpoint, couche et méthode d'extraction sont vérifiés.

Pour chaque paire $(x,y)$, exécuter les ordres $(x,y)$ et $(y,x)$ et, pour chaque ordre, injection dans la première phrase, injection dans la seconde et sham. Apparier les longueurs en tokens ; fixer le séparateur. Croiser les deux mappings de réponse dans toutes les cellules principales, indépendamment de l'ordre et de la condition : mapping présent/absent pour la tâche de présence, et attribution des labels aux positions pour la localisation. Enregistrer la distance du site au token de réponse. Une inversion de l'ordre des phrases ne suffit pas, seule, à distinguer biais pour la lettre et biais pour la position.

**Injection.** Perturber uniquement les tokens du texte de la phrase cible lors du traitement du prompt, à la sortie du bloc retenu. Exclure les marqueurs A/B, séparateurs, consignes et tokens de réponse. Définir les bornes par offsets du tokenizer après application du chat template, sans recherche ambiguë d'une sous-chaîne répétée. Retirer l'intervention pour la génération ; invalider tout cache hérité d'une autre condition. Le sham suit exactement le même chemin d'exécution avec une perturbation nulle.

L'unité de réplication textuelle est la paire, pas chacun de ses swaps. Les concepts et directions sont une seconde source de variabilité, croisée avec les paires. Des répétitions strictement déterministes ne créent pas de nouvelles observations indépendantes.

## Définir une dose commune sans confondre les opérateurs

Soient $H^0_{\ell,T}\in\mathbb{R}^{n\times d}$ les activations sans intervention, sur les $n$ tokens ciblés à la couche $\ell$, et $\Delta_{\ell,T}$ la perturbation effectivement ajoutée. La dose principale est
$$
\rho=\frac{\left\lVert \Delta_{\ell,T}\right\rVert_F}{\left\lVert H^0_{\ell,T}\right\rVert_F+\varepsilon}.
$$
Elle égalise un déplacement relatif euclidien sur le même support. Elle **n'égalise pas** la surprise statistique, l'effet sémantique ou la dégradation aval. Rapporter aussi $\left\lVert \Delta\right\rVert_F/\sqrt n$, la longueur ciblée et les normes par token. Fixer $\varepsilon$ numériquement et signaler les dénominateurs pathologiques.

Pour réaliser une dose cible, générer une matrice de forme $G$, puis poser
$$
\Delta=\rho\,\frac{\left\lVert H^0\right\rVert_F+\varepsilon}{\left\lVert G\right\rVert_F}\,G,
\qquad \left\lVert G\right\rVert_F>0.
$$
Rejeter et journaliser un tirage nul. Ne jamais utiliser le score de détection du test pour régler cette échelle.

- **Concept cohérent :** $G_t=\hat v_c$ pour tous les tokens ciblés, où $\hat v_c$ est une différence de moyennes normalisée extraite sur un corpus indépendant, à cette couche.
- **Aléatoire cohérent :** $G_t=u_{\ell,j}$, avec une direction tirée uniformément sur la sphère séparément pour chaque couche $\ell$, puis maintenue fixe entre tokens et entre phrases à cette couche. Utiliser plusieurs directions, sans sélectionner les plus détectables. Une même suite de coordonnées ne constitue pas une direction comparable entre couches. Ce bras contrôle la cohérence temporelle du steering conceptuel.
- **Bruit renouvelé :** $G_{tk}\sim\mathcal N(0,1)$ indépendamment avant remise à l'échelle du bloc. Il s'agit donc précisément de *bruit gaussien renormalisé en norme de Frobenius* : après normalisation commune, ses coordonnées ne sont plus des gaussiennes indépendantes exactes. Conserver les graines et réutiliser le même tirage de forme dans les comparaisons de doses compatibles.

Une direction gaussienne normalisée fixe a la même loi directionnelle qu'un vecteur isotrope aléatoire. La différence avec le troisième bras vient ici de la corrélation entre tokens. Un écart aléatoire cohérent/bruit renouvelé ne doit donc pas être attribué à la « non-sémantique » de l'un des deux. Une extension utile renouvelle également la direction aléatoire entre essais pour dissocier stabilité entre essais et cohérence entre tokens.

**Grille.** Pilote indépendant avec $\rho\in\{0,0{,}01,0{,}03,0{,}1,0{,}3,1\}$ à titre initial. Pour le noyau, retenir trois doses positives communes aux bras conceptuel et aléatoire, couvrant si possible le début de réponse et le plateau avant effondrement de la tâche. Une quatrième dose et le bras bruit peuvent être ajoutés comme extensions dans une plage commune validée. Enregistrer la règle de sélection ; si aucune plage commune n'existe, le rapporter et ne pas comparer artificiellement des seuils extrapolés.

### Analyses secondaires conditionnelles : géométrie et effet aval

Pour une direction fixe unitaire $v$, estimer sur un corpus témoin au même site et avec le même template
$$
s(\ell,v)=\operatorname{SD}_{h\sim\mathcal C_0}\langle h,v\rangle,
\qquad z_{\ell,v}=\frac{a}{s(\ell,v)},\quad \Delta_t=a v.
$$
Le coefficient $a$ est une norme par token, pas un écart-type gaussien. Stratifier les projections par position si nécessaire ; comparer SD et $1{,}4826\times\mathrm{MAD}$ en robustesse. Les tokens d'une même phrase étant corrélés, estimer l'incertitude par phrases. $z$ mesure le déplacement en unités de dispersion projetée, pas une probabilité d'anomalie multivariée.

Comparer à $z$ égal constitue une **seconde série d'interventions** pour concept et aléatoire cohérent ; ce n'est pas un simple renommage de $\rho$. Pour le bruit par token ou le dropout, la direction change avec le tirage ou l'état : un unique $s(\ell,v)$ n'a pas la même signification. Ne pas présenter une telle extension comme une normalisation universelle.

Si le noyau est terminé, mesurer en complément la divergence JS entre distributions complètes de prochain token sur une continuation neutre fixée et forcée, à des positions fixées après le site. Apparier éventuellement les intensités sur calibration, puis geler la correspondance pour le test et publier la qualité de l'appariement. Ne pas apparier sur les logits du rapport de détection : ils constituent le résultat étudié. L'appariement aval conditionne un effet de l'intervention ; il répond à une autre question que l'effet total à $\rho$ égal et ne prouve pas une médiation causale.

**Dropout en extension.** Définir explicitement $h'=(1-m)\odot h$ avec $m_k\sim\mathrm{Bernoulli}(p)$, sans remise à l'échelle par $1/(1-p)$. Alors $\Delta=-m\odot h$ et $\mathbb E\left\lVert \Delta\right\rVert^2=p\lVert h\rVert^2$ pour $h$ fixé : $\sqrt p$ guide la calibration RMS mais n'est pas la dose réalisée de chaque essai. Cet opérateur diffère de celui de Fornasiere et al., qui appliquent un dropout inversé, remis à l'échelle par $1/(1-p)$, aux sorties de l'attention et du MLP à chaque couche. Il constitue donc une variante contrôlée plutôt qu'une réplication directe. Comparer dans des plages de $\rho$ communes ; renormaliser ensuite $\Delta$ en ferait un autre opérateur, qui ne mettrait plus exactement les coordonnées à zéro.

## Deux tâches principales, évaluées séparément

### Présence/absence : mesurer sensibilité et critère

Utiliser les deux mêmes phrases, avec l'une perturbée ou aucune. Ne donner ni le statut réel de l'essai, ni le nom du concept. Exemple de consigne à figer après pilote :

> An external intervention may or may not have been applied directly to the internal activations while you processed either sentence. Was there a direct activation intervention? Answer with exactly one label: A = present, B = absent.
> Sentence 1: {x}
> Sentence 2: {y}

Ne pas préremplir la réponse correcte. Inverser le mapping présent/absent entre A/B indépendamment des conditions ; vérifier sur un sous-ensemble le format littéral Yes/No. La balance des classes doit être explicite lors du pilote et du test. Avec des shams partagés et moins nombreux, analyser les classes avec poids égaux et ne pas annoncer une prévalence empirique de 50 % dans le prompt.

Utiliser le score $q=\log P(\text{présent})-\log P(\text{absent})$, décodé selon le mapping du prompt. Vérifier la tokenisation des labels dans leur contexte exact ; si plusieurs tokens sont nécessaires, utiliser les log-probabilités des séquences complètes. Rapporter l'AUROC de $q$ contre le statut réellement assigné, sans retourner son signe après observation du test.

Pour la décision discrète $q>0$, noter $H$ le taux de vrais positifs et $F$ le taux de faux positifs. Avec $n_1$ essais perturbés, $n_0$ shams et leurs nombres de réponses « présent » $h,f$, utiliser la correction
$$
\widetilde H=\frac{h+1/2}{n_1+1},\quad
\widetilde F=\frac{f+1/2}{n_0+1},\quad
d'=\Phi^{-1}(\widetilde H)-\Phi^{-1}(\widetilde F),\quad
c=-\tfrac12\bigl[\Phi^{-1}(\widetilde H)+\Phi^{-1}(\widetilde F)\bigr].
$$
Présenter également les taux bruts et les effectifs. L'interprétation de $d'$ et $c$ mobilise le modèle SDT gaussien à variances égales ; l'AUROC empirique complète cette description. Ne pas calculer une courbe ROC informative à partir d'un unique bit lorsque le score continu est disponible.

### Localisation A/B : conserver le signal brut

La consigne demande : « Which sentence was directly targeted? Make a forced choice: A or B. » Même sur les shams, conserver ce format pour mesurer la préférence spontanée ; aucun choix n'y est correct et aucune accuracy de localisation ne doit leur être attribuée. La réussite sur les seuls essais perturbés ne dit pas si le modèle sait reconnaître l'absence d'intervention.

Soit $L=\log P(A)-\log P(B)$. Pour chaque ordre et mapping, orienter $L$ vers la même position physique avant de comparer les essais. Conserver :

- l'accuracy **brute**, moyenne à poids égaux entre les deux sites ; les préférences sur sham et l'asymétrie première/seconde phrase ;
- le contraste expérimental $S=\tfrac12(L_{\mathrm{injection\ A}}-L_{\mathrm{injection\ B}})$, avec A/B référant ici aux sites après remapping ;
- en diagnostic secondaire, $L_{\mathrm{ajusté}}=L-L_{\mathrm{sham}}$ et son accuracy.

L'ajustement utilise une exécution contrefactuelle accessible au chercheur. Il mesure un changement induit par l'intervention et **ne représente pas la décision autonome du modèle**. Il peut accentuer des fluctuations infimes ; l'accompagner des tailles d'effet en logits. Traiter les égalités de scores selon une règle fixée, par exemple demi-crédit en localisation. Ne pas convertir mécaniquement cette accuracy en $d'$ de détection : les deux phrases sont traitées dans une séquence causale commune, pas dans deux canaux indépendants.

## Contrôles nécessaires à l'interprétation

**Sham :** intervention exactement nulle. Une direction aléatoire non nulle est un contrôle actif. Vérifier à dose zéro l'égalité des sorties avec/sans hook à la tolérance numérique fixée. Un sham identique peut être réutilisé pour plusieurs contrastes, mais ne doit pas être compté plusieurs fois dans l'effectif indépendant.

**Biais de réponse :** mappings inversés, ordre des phrases, format Yes/No contre labels neutres et quelques questions factuelles dont les réponses oui/non sont connues et équilibrées. Un effet sur ces questions indique qu'une variation de logits n'est pas spécifique au rapport introspectif. Le choix forcé réduit certains biais sans les supprimer tous. Conserver aussi les probabilités attribuées hors des labels autorisés et vérifier le taux de réponses valides sur un sous-ensemble généré librement ; une accuracy conditionnelle aux labels masque sinon un défaut de suivi de consigne.

**Contrôle textuel de type gaslight :** trois conditions assignées : injection interne, induction du concept par une phrase de consigne sans hook actif, et sham. Demander de classer la provenance parmi interne / texte / aucune, avec labels permutés. Calibrer la force de l'induction textuelle sur des mesures aval indépendantes de la détection ; si l'appariement échoue, le dire. Faire un pilote puis un test sur phrases et paraphrases nouvelles ; ne pas attribuer l'origine à partir d'un simple marqueur textuel fixe.

Ce contrôle est **asymétrique** : une fausse attribution d'injection au texte réfute la spécificité dans cette condition ; une classification correcte peut être obtenue par lecture de la consigne visible. Elle ne démontre donc pas un accès privilégié. Ajouter, sur le même prompt contenant une suggestion textuelle fixe, une randomisation cachée injection réelle/sham permet de tester une information au-delà des seules entrées. Cela ne prouve toujours pas un calcul de second ordre. Un observateur ne recevant que le prompt permet d'estimer la facilité de classification par les indices visibles ; ne pas lui fournir un rapport qui révélerait la condition.

**Dégradation :** poser dans une exécution séparée une question de compréhension à réponse connue sur la phrase cible, avec la même intervention et le même préfixe de texte. Apparier des choix neutres pour éviter le concept dans la réponse. Mesurer accuracy et variation de log-probabilité de la réponse correcte par rapport au sham, sans changer de question selon le résultat. Publier toutes les doses ; annoter les zones où la perte dépasse, par exemple, 5 points, seuil pratique fixé avant test. Détecter avant une dégradation importante est intéressant ; cela n'exclut pas des indices internes de perturbation plus subtils.

## Courbes, incertitudes et charge de calcul

**Fonction psychométrique.** Pour la localisation, ajuster si les données sont compatibles une sigmoïde bornée avec plancher $0{,}5$ et asymptote $1-\lambda$. Définir $\rho_{75}=\inf\{\rho:p_{\mathrm{correct}}(\rho)\geq0{,}75\}$. Ne rapporter un seuil que si le domaine observé encadre ce passage. Si la performance chute aux fortes doses, montrer les points et restreindre explicitement l'ajustement à la branche croissante préspecifiée, ou renoncer au résumé par seuil. Pour la détection, rapporter TPR en fonction de la dose avec FPR et AUROC ; son plancher n'est pas automatiquement 0,5.

**Incertitudes.** Intervalles à 95 % par bootstrap croisé sur paires et concepts/directions, en gardant ensemble tous les swaps, doses et shams associés. Conserver le partage des shams dans chaque rééchantillonnage. Pour le critère principal, calculer dans chaque rééchantillonnage les deux AUROC puis leur différence, en n'incluant qu'une fois les shams partagés : l'intervalle porte ainsi directement sur le contraste apparié. Avec seulement six grappes conceptuelles, le bootstrap au niveau des concepts reste instable : rapporter les résultats par concept, ajouter une analyse de sensibilité *leave-one-concept-out* et qualifier l'inférence comme exploratoire hors de cet ensemble. Un modèle mixte avec effets de paire et de concept peut compléter les contrastes appariés, sans remplacer les données brutes. Les IC ponctuels des courbes ne sont pas des bandes simultanées ; appliquer une correction de multiplicité aux familles de tests secondaires revendiquées.

Une différence non significative n'est pas une équivalence. Pour soutenir une équivalence pratique, fixer une marge à l'avance, par exemple $\pm0{,}05$ d'AUROC, et vérifier que l'intervalle pertinent est entièrement inclus dans cette marge. Sinon conclure « différence non résolue ». Les seuils non atteints sont censurés, pas imputés à la dose maximale.

**Budget du noyau confirmatoire.** Par tâche, le plan comprend
$$
40\ \text{paires}\times6\ \text{directions}\times1\ \text{couche}\times3\ \text{doses}
\times2\ \text{ordres}\times2\ \text{sites}\times2\ \text{familles}\times2\ \text{mappings}
=11\,520
$$
exécutions perturbées, soit 23 040 pour présence et localisation. Ajouter 160 shams par tâche, correspondant à 40 paires, deux ordres et deux mappings, ainsi que les exécutions de validation des hooks et le sous-plan de compréhension. L'ajout de la seconde couche double ce noyau à 46 080 exécutions perturbées. Le plan complet avec trois familles, deux couches, quatre doses et deux mappings atteindrait 92 160 exécutions perturbées avant pilote et contrôles ; il n'est donc pas compatible avec un plafond de 60 000. Le bruit, la quatrième dose, la seconde couche et les autres extensions doivent être ajoutés séparément après chronométrage, sans dépasser une enveloppe gelée au terme du pilote. Dans le bras bruit, les six index désignent des graines, pas des catégories sémantiques.

La faisabilité dépend du GPU, de la longueur de prompt, des batches et des passes témoins nécessaires ; aucun temps d'exécution n'est garanti ici. Chronométrer au jour 1 un lot représentatif incluant shams et journaux. Si le budget est dépassé, réduire d'abord les extensions et le nombre de doses, en conservant les shams et les contrôles. Ne pas traiter les dizaines de milliers d'exécutions comme autant de phrases indépendantes : le plan n'en contient que 40 paires.

## Diagnostic causal et extensions conditionnelles

### Propagation entre phrases et restauration

La causalité autorégressive rend première et seconde phrase asymétriques. Conserver le diagnostic
$$
P_{1\to2}(\ell)=\frac{\left\lVert H^{(1)}_{\ell,2}-H^{(0)}_{\ell,2}\right\rVert_F}{\left\lVert H^{(0)}_{\ell,2}\right\rVert_F+\varepsilon},
$$
sur un sous-ensemble, avec comparaison inverse $P_{2\to1}$ comme contrôle de l'implémentation. À la sortie exacte du bloc où seuls les tokens de la première phrase sont modifiés, les tokens de la seconde ne changent pas encore ; la propagation peut apparaître dans les blocs suivants. Un effet rétrograde, au-delà des tolérances numériques, doit faire rechercher une erreur de masque, de hook ou de cache.

Pour tester une contribution de la seconde phrase au score, restaurer ses activations depuis le sham à un site aval explicite et recalculer tous les descendants, y compris les caches affectés. Comparer injection seule, injection avec restauration, sham et restauration factice. Le contraste $E=L_{\mathrm{injection}}-L_{\mathrm{restauration}}$ mesure l'effet causal de cette restauration dans ce dispositif, pas un « pourcentage d'introspection » : les interactions et la mise hors distribution empêchent une décomposition simple.

Une sonde hors échantillon établit qu'une information est décodable par le chercheur ; elle n'établit pas que le modèle l'utilise. Une piste vers le second ordre serait d'identifier un composant candidat au rapport, puis de tester sa nécessité et sa suffisance tout en préservant la représentation du concept, la tâche et les capacités générales de réponse. Cela dépasse le noyau du hackathon ; un simple logit lens ou une corrélation sonde/rapport n'y suffit pas.

### Multi-injection : priorité à un test d'interaction

Remplacer la série ambitieuse de comptage, identité et ordre de profondeur par un premier plan $2\times2$ : aucune injection, A seule, B seule, A+B. Fixer les tokens, contrôler les deux couches et calibrer chaque composante sur la trajectoire propre ; enregistrer aussi sa dose sur l'état effectivement atteint. Comparer dose par composante constante et budget nominal total constant, par exemple $\rho_A^2+\rho_B^2$ constant. La somme de doses à des couches différentes est un budget conventionnel, pas la norme d'une perturbation unique.

L'interaction $I=q_{AB}-q_A-q_B+q_0$ caractérise la non-additivité sur l'échelle du score choisie. Même un $I$ non nul ne montre ni perception de deux événements ni connaissance de l'architecture.

Si le comptage est ensuite testé, dissocier nombre de hooks, nombre de couches et nombre de concepts distincts : injecter deux fois le même concept ne constitue pas deux contenus distincts. Inclure des essais à énergie globale contrôlée ; ne pas annoncer le nombre dans une consigne qui prétend l'évaluer. L'identification avec nombre annoncé est une autre tâche, admissible mais conditionnelle à cette information.

Pour l'ordre de profondeur, croiser A précoce/B tardif avec B précoce/A tardif, les identités, le mapping des réponses et les intensités relatives ; garder le site textuel identique. Inclure une condition même couche comme diagnostic sans réponse correcte d'ordre. Un succès peut provenir d'une différence de saillance ou de transformation selon la profondeur ; il n'établit pas une connaissance explicite du numéro de couche.

La similarité cosinus de vecteurs doit être calculée dans un espace *à couche fixée*. Comparer directement des vecteurs provenant de couches différentes, même de même dimension, ne fournit pas une mesure sémantique interprétable sans alignement justifié. La similarité sémantique et la similarité géométrique doivent être évaluées séparément.

### Autres pistes à conserver

- Comparer plusieurs estimateurs du même concept sur des corpus d'extraction disjoints. Mesurer la stabilité des directions, pas seulement le meilleur taux de détection.
- Répliquer le noyau sur un second modèle seulement après validation du premier. Deux modèles différents ne suffisent pas à isoler un effet architectural.
- Comparer base/instruct avec contrôle textuel si le temps le permet, en vérifiant la compréhension des consignes. Seuls des checkpoints successifs appropriés permettent de discuter spécifiquement l'effet de DPO.
- Mesurer une confiance explicitement demandée sur une tâche définie. Le meta-$d'$ nécessite suffisamment de réponses correctes et incorrectes et des niveaux de confiance exploitables ; le rapport meta-$d'/d'$ devient instable lorsque $d'$ est proche de zéro. Il ne prouve pas un mécanisme métacognitif. Les log-probabilités des labels ne sont pas une confiance verbalisée.

## Décisions, calendrier et livrables

| **Jours** | **Travail** | **Critère de passage** |
| --- | --- | --- |
| 1--2 | Valider hooks, tokenisation, shams, précision et corpus ; chronométrer ; répliquer un signal de référence | Dose nulle identique au témoin ; support d'injection exact |
| 3 | Pilote indépendant : plage de doses, compréhension des consignes, validité des labels | Geler doses, contrastes, exclusions et budget |
| 4--6 | Noyau présence/localisation sur le test ; compréhension sur sous-plan fixé | Journaux complets et toutes les cellules du noyau |
| 7--8 | Courbes, IC, effets de position ; seconde couche, bruit ou contrôle textuel selon le budget gelé | Conclusions avec incertitudes et limites |
| 9 | Réplication ciblée ou multi-injection $2\times2$, uniquement si le noyau est terminé | Extension clairement séparée |
| 10 | Relecture, figures et rédaction ; archiver prompts, graines et données | Résultats reproductibles, y compris nuls |

**Journal minimal par essai :** identifiants de paire, concept/direction et split ; modèle/révision/précision ; prompt exact et mapping ; couche et site ; bornes tokens ; opérateur et graine ; dose visée et réalisée ; score des labels et décision ; validité du format ; identifiant du sham partagé ; résultats de compréhension et JS si mesurés. Sauvegarder les activations complètes seulement pour le sous-ensemble mécaniste.

**Figures principales :** AUROC et TPR/FPR selon $\rho$ ; localisation brute selon $\rho$ avec IC ; dégradation selon $\rho$ ; contrastes concept/aléatoire par concept. Ajouter les résultats par couche et la matrice interne/texte/sham si les extensions correspondantes sont exécutées, ainsi que les diagnostics ajustés en annexe avec leurs scores bruts.

**Règles d'interprétation :** un avantage conceptuel persistant indique une différence inexpliquée par les contrôles retenus ; sa disparition après normalisation est compatible avec une explication d'échelle, sans prouver que toute la géométrie est contrôlée ; une absence de différence reste indéterminée hors d'un test d'équivalence ; une confusion texte/interne limite la spécificité ; un succès limité aux fortes dégradations évoque un signal de dysfonctionnement. Aucun de ces seuls résultats ne démontre une représentation de second ordre.

## Références méthodologiques

Les neuf articles ML, leurs versions et la matrice du corpus figurent dans la synthèse révisée jointe. Les sources directement structurantes pour ce protocole sont :

- Hahami et al., tâches et biais de réponse : [https://arxiv.org/abs/2512.12411v2](https://arxiv.org/abs/2512.12411v2).
- Fornasiere et al., dropout/bruit et doses : [https://arxiv.org/abs/2604.17465v2](https://arxiv.org/abs/2604.17465v2).
- Singh et al., contrôle textuel et second ordre : [https://arxiv.org/abs/2605.26242v2](https://arxiv.org/abs/2605.26242v2).
- Ferrara, sham, rapports et confiance : [https://arxiv.org/abs/2608.20569v1](https://arxiv.org/abs/2608.20569v1).
- Fleming et Lau (2014), *How to measure metacognition* : [https://doi.org/10.3389/fnhum.2014.00443](https://doi.org/10.3389/fnhum.2014.00443).
- Maniscalco et Lau (2012), estimation du meta-$d'$ : [https://doi.org/10.1016/j.concog.2011.09.021](https://doi.org/10.1016/j.concog.2011.09.021).
