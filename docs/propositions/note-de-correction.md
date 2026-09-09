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
