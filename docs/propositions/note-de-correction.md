# Proposition de correction scientifique

9 septembre 2026 — Lecture de `docs/reponse_synthese_bilio.md`, de la synthèse LaTeX et des huit pages du cadrage initial. Vérification ciblée des neuf articles du corpus et de deux références de métacognition. Les versions proposées sont séparées des originaux. Aucun résultat expérimental nouveau n'est revendiqué et le code d'expérimentation n'a pas été modifié.

## Avis de chercheur ML

Le projet est pertinent à condition de déplacer sa contribution : **mesurer ce qui rend une perturbation rapportable dans un protocole commun**, puis tester certaines explications concurrentes. Démontrer une introspection forte serait une ambition beaucoup plus exigeante, qui demanderait de montrer l'usage causal d'une représentation de second ordre.

Je recommande de conserver comme livrable principal une comparaison concept/aléatoire cohérent/bruit, assortie de shams, de courbes dose-réponse et d'une mesure de fonctionnement. La multi-injection est une extension intéressante, mais elle ne doit pas conditionner la réussite du projet.

Les documents proposés sont :

- `synthese-bibliographique-introspection-llm.tex` : texte intégral révisé, matrice comparative et bibliographie avec versions explicites.
- `cadrage-experiments.tex` : remplacement rédigé du cadrage, avec hypothèses, dose, tâches, contrôles, métriques, budget et calendrier.
- Les deux PDF correspondants se trouvent dans `output/pdf/` à la racine du projet.

## 1. Corrections de la synthèse, dans l'ordre de priorité

| Passage de la version initiale | Correction proposée | Raison |
|---|---|---|
| Macar : capacité présente mais peu stimulée dans les modèles de base | Distinguer absence du mécanisme dans les modèles de base étudiés et sous-élicitation dans les modèles post-entraînés | Inversion du résultat ; éviter de transformer le résultat DPO en loi générale |
| Gains de 53 % et 75 %, sans faux positifs significatifs | Points de pourcentage ; donner les taux bruts, les faux positifs et la contrepartie sur prefill | Une amélioration de détection ne suffit pas si le critère de réponse se déplace |
| Kowalski : introspection et capacité incertaine sans fine-tuning | Contrôle d'activations sur instruction, étudié sans nouvel entraînement pour le benchmark | Contrôle, détection et connaissance de l'état sont distincts |
| Comportement malveillant caché dans les poids | Retirer cette extrapolation | Ni la cible expérimentale ni le dispositif ne démontrent cela |
| Mishra : aucun prompt ne reproduit le comportement | Égalité exacte d'états internes, sous hypothèses | Des états différents peuvent produire le même comportement |
| Mishra soutiendrait une détection d'anomalies plutôt qu'une introspection | Présenter ce rapprochement comme une hypothèse du projet | Le papier ne teste pas l'introspection |
| Ferrara : perturbation conceptuelle, généralisation parfaite, simple révélation d'un signal | Opérateurs non conceptuels ; zéro erreur observée dans un test borné ; mécanisme du fine-tuning non établi | Ne pas confondre résultat fini, généralisation et explication mécaniste |
| Singh : deux principes conceptuels, entrées sans anomalie | Mettre au centre l'expérience gaslight et la confusion entre origines | L'anomalie d'entrée est un contrôle actif essentiel |
| Hahami décrit comme un ensemble de tâches 2AFC | Séparer oui/non, comparaison d'intensité et localisation parmi dix phrases | Les chances de base et les capacités mesurées diffèrent |
| Fornasiere : classification décevante ; intensité à ajouter | Séparer localisation, classification zero-shot et gains avec exemples ; reconnaître les balayages existants | La proposition initiale sous-estime le résultat et revendique un prolongement déjà réalisé |
| Tous les facteurs étudiés séparément ; localisation peu explorée | Décrire les combinaisons dans une matrice ; distinguer localisation de phrase et identification de couche | La nouveauté doit porter sur les contrôles et la comparaison commune |
| Différences de profondeur attribuables à l'architecture | Présenter plusieurs explications : tâche, dose, site, temporalité, modèle, entraînement | Une comparaison entre articles ne permet pas une attribution causale à l'architecture |

### Deux nuances au fichier de remarques

**Fornasiere :** la formule « aucun sham » est trop absolue. Le papier inclut une dose nulle en localisation, explicitement décrite comme revenant au hasard. Ce qui manque à cette tâche est la possibilité d'exprimer « aucune intervention » et donc de mesurer les fausses déclarations de présence. Il faut critiquer la grandeur non mesurée, pas l'absence de tout témoin. Voir [§4.1 du papier](https://arxiv.org/pdf/2604.17465), p. 4 du PDF.

**Ferrara :** ne pas figer sans préciser la batterie le nombre de modèles avec contrôles appariés. Le texte consulté distingue contrôles de norme unitaire, calibrations de dose et nouvelles grilles non utilisées dans les résultats rapportés ; l'annexe A mentionne notamment deux modèles calibrés. La correction robuste est : « l'appariement aval n'est pas systématiquement réalisé sur les huit modèles ». Voir [annexe A, tableau de couverture exécutée](https://arxiv.org/html/2608.20569v1#A1). Il faut citer le dispositif effectivement analysé, pas simplement la disponibilité d'un mécanisme de calibration.

## 2. Corrections du cadrage, page par page

| Emplacement initial | À conserver | À modifier |
|---|---|---|
| p. 1, liste d'idées | Comparaison de perturbations et normalisations ; variabilité des directions | Remplacer l'inventaire ouvert par cinq hypothèses, une cible principale et des extensions |
| p. 2, tâche A/B et six exécutions | Deux ordres, deux sites et contrôle nul | La nommer localisation ; ajouter une tâche distincte de présence/absence et croiser le mapping des labels |
| p. 3, logits ajustés | Score brut, différence avec sham et contraste de site | L'accuracy ajustée est une analyse contrefactuelle du chercheur, pas la performance autonome du modèle |
| p. 3, propagation et restauration | Analyse de l'asymétrie autorégressive | Fixer le site et recalculer les descendants ; distinguer propagation, restauration causale et mécanisme de second ordre |
| pp. 3–4, estimation de s(l,v) | Calibration sur corpus indépendant et robustesse SD/MAD | Réserver la dispersion projetée aux directions fixes ; ne pas confondre coefficient gaussien, norme et dose réalisée |
| p. 4, tableau d'interprétations | Comparaisons à norme commune puis selon géométrie | Transformer les conclusions causales en interprétations conditionnelles ; « pas de différence » ne signifie pas équivalence |
| p. 5 | — | Supprimer la page vide |
| p. 6, comptage et identification | Matrices de confusion et scoring indépendant | Séparer nombre d'événements et nombre de concepts ; une injection aléatoire n'est pas un sham |
| p. 7, profondeur et similarité | Permutations, intensités, couches et concepts | Contrôler la dose totale et les effets de saillance ; cosinus à couche fixée ; même couche sans ordre correct |
| p. 8, sonde/logit lens | Décodabilité hors échantillon et comparaison avec le rapport | Retirer « the only way » ; une sonde n'établit pas l'usage causal du signal par le modèle |

## 3. Ajouts méthodologiques importants

### La question centrale est l'identifiabilité des explications

Même une meilleure détection des concepts à dose égale n'identifie pas un accès sémantique introspectif. Les directions peuvent différer par leur alignement avec les circuits de réponse, leur cohérence entre tokens, leur facilité de verbalisation ou leur impact sur le fonctionnement. Le protocole doit d'abord contrôler les variables manipulables et annoncer ce qui reste indéterminé.

Le contraste **concept contre aléatoire cohérent** est le plus propre pour le noyau : même support, même dose et même direction répétée sur les tokens. La comparaison avec un bruit renouvelé teste aussi la cohérence temporelle. Une direction gaussienne normalisée fixe n'est pas une quatrième famille distincte d'une direction isotrope aléatoire.

### Ne pas appeler trois quantités différentes « z »

- **ρ**, norme relative de la perturbation sur le support ciblé : dose principale, applicable à des opérateurs différents.
- **z directionnel**, amplitude divisée par la dispersion des projections naturelles : analyse secondaire de directions fixes.
- **JS aval**, divergence des distributions de sortie sur une continuation indépendante du rapport : mesure d'impact, puis appariement secondaire éventuel.

Le coût **J** de Nguyen constitue encore une autre grandeur. Aucune équivalence entre ces mesures n'est supposée. Un appariement JS ne doit pas utiliser le score même que l'on veut comparer et ne devient pas, par son existence, une preuve de mécanisme causal.

### Deux réserves statistiques à rendre explicites

Un sham réutilisé dans cent contrastes reste un seul témoin. Les swaps, concepts et doses d'une même paire sont dépendants : bootstrap par unités appropriées, pas test binomial naïf sur toutes les exécutions. Dans le plan proposé, 46 080 passes perturbées correspondent seulement à 40 paires et six concepts/directions par famille.

Le seuil à 75 % n'est défini que si la courbe atteint ce niveau dans le domaine mesuré. Une chute à forte dose invalide un résumé monotone global. Une différence non significative ne permet de conclure ni que toutes les perturbations sont équivalentes ni que seule l'anomalie compte.

### Ce que le contrôle gaslight peut et ne peut pas établir

Une manipulation textuelle attribuée à tort à une injection limite la spécificité du rapport. Mais reconnaître correctement une consigne textuelle visible peut être trivial. D'où l'ajout proposé : à prompt textuel identique, randomiser secrètement injection/sham ; l'entrée seule ne prédit alors pas le statut de l'injection. Un succès apporte une preuve comportementale plus discriminante, tout en restant insuffisant pour une représentation de second ordre. Cette distinction suit la motivation de [Singh et al.](https://arxiv.org/html/2605.26242v2).

### Lecture ponctuelle du code existant

Le helper `code/utils/inject_concept_vector.py` propose notamment une injection depuis un token jusqu'à la fin et pendant la génération. Cette portée ne réalise pas automatiquement la tâche ciblée sur une seule phrase. En revanche, `code/experiments/position_detection.py` annonce un masque de phrase avec génération propre. Le cadrage révisé impose un contrat de support exact et sa vérification ; il ne suppose pas que tous les scripts utilisent le même chemin. Ce constat est une lecture ciblée, pas un audit complet ni un test d'exécution du code.

## 4. Priorisation des idées de l'équipe

| Idée | Recommandation |
|---|---|
| Thomas : effet base/post-training avec contrôle Singh | Extension après validation du noyau ; contrôler le suivi des consignes et utiliser des checkpoints appropriés si la question est spécifiquement DPO |
| William : éviter le biais oui/non et tester les normalisations | Noyau : garder présence/absence avec mappings contrôlés et localisation séparée ; comparer le format littéral oui/non sur un sous-plan |
| Seif : plusieurs vecteurs du même concept | Bonne extension ; dissocier stabilité de l'estimation et sélection opportuniste du vecteur le plus détectable |
| Louis : multi-injection et comparaison relative | Commencer par aucune/A/B/A+B, puis tester l'ordre seulement si l'identification simple fonctionne |
| Améliorer les tâches annexes ou étudier des modèles reasoning | Mesurer d'abord la dégradation d'une tâche contrôlée ; amélioration et modèles reasoning demandent un projet ou une réplication supplémentaire |

## 5. Sources et portée de la vérification

Les sources ont été consultées pour les affirmations décisives ; il ne s'agit pas d'une recherche systématique exhaustive. La correction s'appuie également sur les détails fournis dans le fichier de remarques. Les chiffres de performances restent attachés aux protocoles des auteurs et ne prédisent pas les résultats du projet.

| Source consultée | Sections utiles |
|---|---|
| [Hahami, v2](https://arxiv.org/html/2512.12411v2) | §§4–6 : biais, tâches et profondeur |
| [Macar, v2](https://arxiv.org/html/2603.21396v2) | §§3.3, 5–6 : post-entraînement, circuit et amplification |
| [Fornasiere, v2, PDF](https://arxiv.org/pdf/2604.17465v2) | §§4–5 : localisation, calibration et classification |
| [Lindsey, v1](https://arxiv.org/html/2601.01828v1) | Critères, distinction entrée/concept, tendances et limites |
| [Singh, v2](https://arxiv.org/html/2605.26242v2) | §§3–4 et appendices : accès privilégié, gaslight et trois conditions |
| [Mishra, v2](https://arxiv.org/html/2604.09839v2) | Théorème, expériences de prompting et limites de quantification |
| [Nguyen, v1](https://arxiv.org/html/2605.01167v1) | Coût géométrique et analyse performance/dommage |
| [Kowalski, v1](https://arxiv.org/html/2608.21664v1) | Contrôle sur instruction, checkpoints et limites des concepts |
| [Ferrara, v1](https://arxiv.org/html/2608.20569v1) | §7, contrôle positif, confiance et annexe de couverture |
| [Fleming et Lau, 2014](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00443/full) | Sensibilité, biais et efficacité métacognitive |
| [Maniscalco et Lau, 2012](https://brianmaniscalco.org/wp-content/uploads/2018/10/Maniscalco-Lau-2012-Consc-Cog-corrected.pdf) | Définition et estimation du meta-d' |

Deux références psychophysiques ciblées ont été ajoutées à la synthèse pour limiter son allongement. Green & Swets (1966) et Macmillan & Creelman (2005) restent de bons compléments de fond si le rapport final développe davantage la théorie de la détection du signal.

## Vérification de la version Markdown après les remarques — 9 septembre 2026

Périmètre : `docs/propositions/reponse_synthese_bilio.md` confronté à `docs/propositions/synthese-bibliographique-corrigee.md`. La référence Maniscalco et Lau avait été omise dans le Markdown ; elle est maintenant ajoutée sous le numéro **[11]**, citée dans le texte et accompagnée d’une définition du meta-d’. Les dix paragraphes initiaux et les neuf lignes du tableau de ressources restent en place. Une matrice complémentaire répond à la suggestion jusque-là non intégrée.

| Remarque | Traitement dans la synthèse corrigée |
| --- | --- |
| Macar : base/instruct inversés ; portée de DPO | § de texte 3 : distinction rétablie, contraste TPR − FPR de l’expérience DPO/SFT précisé, sans généralisation universelle. |
| Macar : points, faux positifs, prefill | §3 : +53 et +75 points ; FPR 0 → 7,3 % et 0 % ; prefill 36 → 16 %. Modèle, couche et coefficient de l’abliteration précisés. |
| Macar : variabilité des concepts et norme | Ligne Macar du tableau ; variabilité entre concepts/directions intégrée au projet (§10). |
| Kowalski : sans nouvel entraînement ; contrôle distinct de détection | §2–4 et ligne Kowalski ; progression du score explicitée sur la trajectoire OLMo 7B, sans singulariser DPO. |
| Kowalski : malveillance et poids | Extrapolation retirée ; concepts simples et limites d’inférence précisés. |
| Mishra : égalité d’états ≠ équivalence de comportement | §2 et ligne Mishra ; exemples de prompting mentionnés ; rapprochement avec l’anomalie présenté comme hypothèse du projet (§8). |
| Mishra : quantification et affiliation | Ligne Mishra : théorie non étendue à la quantification malgré le test INT4 ; Johns Hopkins University. |
| Ferrara : opérateurs non conceptuels, JS | §3, ligne Ferrara et matrice ; couverture partielle de l’appariement explicitée, avec la réserve documentaire ci-dessous. |
| Ferrara : généralisation et mécanisme | Zéro erreur observée sur l’échantillon, borne d’environ 3 %, site et tokens du test précisés ; absence de preuve que le fine-tuning ne fait que révéler une lecture existante. |
| Ferrara : auteur unique et confiance | Auteur au singulier ; AUROC discrète 0,500 contre confiance 0,647, sans équivalence avec une preuve de métacognition. |
| Singh : expériences, gaslight, trois conditions | §8 et ligne Singh : manipulation textuelle comme contrôle actif ; confusion d’origine et Llama-70B près du hasard. Le résumé ne réduit plus l’article à deux principes conceptuels. |
| Hahami : trois tâches et niveaux de hasard | §2 et ligne Hahami : oui/non, comparaison de deux intensités, localisation parmi dix phrases ; 83/88 % et chances de 50/10 %. |
| Hahami : L0–L5 versus signal tardif | §4 : réussite du rapport distinguée du signal des têtes ; 59 % des 1 024 têtes dépassent le hasard dans l’analyse. |
| Fornasiere : localisation, classification, contexte | §6 et ligne Fornasiere : résultats séparés ; intensités suffisantes, modèles et gains avec exemples précisés. |
| Fornasiere : balayages déjà réalisés | §6 : pmin/pmax et grille p × σ ; revendication de nouveauté de l’intensité retirée. |
| Lindsey : tous les modèles, profondeur, quatrième critère | §2 et §4 ; ligne Lindsey : portée de la tâche au-dessus du hasard et critère métacognitif non démontré. |
| Nguyen : corrélation locale et métrique distincte | Ligne Nguyen : Qwen2.5-14B-Instruct, direction/balayage étudiés ; aucune loi sur la détectabilité. |
| Facteurs déjà croisés et nouveauté | §9 et nouvelle matrice types × doses × couches × modèles × contrôles × réponses. |
| Localisation de phrase ≠ identification de couche | §4 et §10 ; multi-injection et couche restent des extensions, sans revendiquer la nouveauté de la localisation de phrase. |
| Divergences du corpus ≠ contradictions | §9 : tâches non comparables directement ; Hahami/Macar explicitement distingués et explications concurrentes énumérées. |
| Sham, FPR, formats et biais | §10 : FPR, oui/non et choix forcé, permutations, absence de préremplissage affirmatif ; nuance sur Fornasiere dans son tableau. |
| Sensibilité, critère, courbes et incertitudes | §10 : d’, c, AUROC sur score continu et fonctions psychométriques. |
| Dose commune, géométrie, sorties | §10 : norme relative comme dose principale, distincte de J et de la divergence aval. |
| Variabilité, dégradation et confiance | §3 et §10 : métriques distinctes ; comparaison entre concepts/directions et tâche témoin. |
| Psychophysique et meta-d’ | [10] Fleming et Lau ; [11] Maniscalco et Lau, avec définition, objet et limites du meta-d’. Les deux ouvrages Green & Swets et Macmillan & Creelman étaient proposés comme ajouts éventuels ; ils restent facultatifs pour éviter un élargissement bibliographique supplémentaire. |
| Forme | Affiliation corrigée, Ferrara au singulier, identifiants bibliographiques redondants supprimés ; numéros [1]–[10] conservés. |
| Faisabilité | §10 : Llama 8B, couche précoce/médiane, trois familles, extensions ; le protocole détaillé et le calendrier restent dans le cadrage. |

### Réserves : prendre une remarque en compte ne signifie pas la recopier sans nuance

- **Fornasiere « aucun sham »** : la condition p = σ = 0 existe en localisation. La limite correcte est l’absence de mesure de FPR de présence dans cette tâche de localisation forcée. Cette nuance est conservée, plutôt que l’affirmation absolue.
- **Ferrara « un modèle sur huit »** : le texte consulté distingue la batterie principale et les modèles calibrés ; son annexe de couverture mentionne des normes calibrées pour deux modèles. La synthèse conserve la formulation vérifiable « appariement non systématique sur les huit modèles ». Voir [l’annexe de couverture](https://arxiv.org/html/2608.20569v1#A1).
- **Hahami « préremplit Yes »** : ne pas étendre ce point à toutes les tâches. La détection binaire décrite au §4.1 compare les logits YES/NO au premier token de génération. Le risque de réponse imposée est traité dans notre protocole en excluant tout préremplissage affirmatif ; la synthèse ne lui attribue pas une procédure unique pour ses trois tâches. Voir [Hahami, §4.1 et annexes des prompts](https://arxiv.org/html/2512.12411v2).

### Provenance du cadrage avant/après

- `cadrage-experiments-avant.md` transcrit le PDF initial de huit pages présent dans ce dossier.
- `cadrage-experiments-apres.md` convertit le LaTeX révisé présent dans ce dossier, sans nouvelle modification de fond. Il comporte déjà Maniscalco et Lau dans les références méthodologiques.

Les problèmes scientifiques volontairement conservés dans le fichier « avant » ne constituent pas des recommandations. Le `.tex` et le PDF sources ne sont pas remplacés par ces conversions.
