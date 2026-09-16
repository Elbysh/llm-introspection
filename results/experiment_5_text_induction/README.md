# Expérience 5 — induction textuelle et fausses déclarations d'injection

Résultats de la section 9 de `docs/livrables/cadrage-experiments.md` (hypothèse
H5). Seule la campagne rapportée dans l'article est publiée ici : les trois
séries exploratoires qui l'ont précédée sont reproductibles à partir des
configurations versionnées dans `configs/experiment_5_text_induction/`.

## Périmètre

- modèle : `meta-llama/Llama-3.1-8B-Instruct`, `dtype` bfloat16 ;
- configuration : `configs/experiment_5_text_induction/induction_early_band.yaml` ;
- 10 concepts, 100 paires textuelles minimales, 163 phrases distinctes ;
- blocs décodeur 0 à 13, amplitude brute `alpha` dans {128, 256} ;
- 44 800 essais, moitié sham et moitié injectés.

Le plan croise les quatre conditions de la section 9.3 : phrase cible peu
évocatrice ou évocatrice, croisée avec sham ou injection du même concept. Les
deux mappings X/Y et les deux positions A/B sont équilibrés, chaque essai
s'exécute dans un contexte neuf, et les deux versions d'une paire reçoivent la
même direction et la même amplitude, sans recalibrage sur le texte évocateur.

## Contenu

| Fichier | Description |
| --- | --- |
| `induction_early_band/trials.jsonl` | un enregistrement par essai (LFS) |
| `induction_early_band/pairs.json` | les 100 paires retenues et leurs empreintes |
| `induction_early_band/conditions.json` | directions, couches et amplitudes |
| `induction_early_band/run_manifest.json` | configuration et empreintes des sources |
| `induction_early_band_analysis/presence_summary.csv` | taux et marges par condition |
| `induction_early_band_analysis/paired_text_effects.csv` | effets appariés et intervalles |
| `induction_early_band_analysis/mapping_and_position.csv` | effets de mapping et de position |
| `induction_early_band_analysis/induction_depth.png` | figure de l'article (LFS) |

Les prompts rendus et les 140 figures par condition ne sont pas versionnés :
ils se reconstruisent à partir de `pairs.json` et de `presence_summary.csv`.

## Résultats

La mesure principale reste au plancher : **0 faux positif sur 800 prompts sham
distincts** dans les deux versions du texte, soit une borne supérieure de 0,38 %
à 95 % (règle de trois). Le texte évocateur déplace néanmoins la marge de
présence de **+0,485** logit, intervalle `[+0,066, +0,827]` en rééchantillonnant
conjointement concepts et phrases. Les deux constats tiennent ensemble parce que
le modèle se tient à −3,36 logits du seuil de décision.

L'effet de l'injection **change de signe avec la profondeur** : −0,769
`[−0,819, −0,719]` sur les blocs 1 à 5, puis +0,454 `[+0,401, +0,507]` sur les
blocs 6 à 13. Agréger toute la bande renvoie −0,089, moyenne des deux régimes
qui ne décrit aucun bloc. L'effet du texte, lui, est positif à chaque bloc et
s'agrège à +0,208 `[+0,197, +0,220]`.

## Précautions de lecture

- **Bloc 0 à écarter.** 88 % de ses essais injectés ne produisent aucune réponse
  valide : `alpha = 128` y vaut environ 108 SD, contre 0,7 à 1,9 SD ailleurs. Sa
  variation de marge est un échec de décodage, pas une déclaration de présence.
- Les blocs 3, 10, 11 et 12 dépassent 15 % de réponses invalides et sont exclus
  des agrégats.
- **Ne jamais agréger l'effet d'injection sur la bande** sans vérifier le signe.
- Le bloc 31 est absent de toutes les configurations : ses directions de concept
  exigent `data/saved_vectors/llama/*_32_avg.pt`, non versionné, et un vecteur
  régénéré ne retrouve pas la norme enregistrée par la calibration.
- La calibration consommée est `isolated_sentence_development` ; les
  configurations utilisent `mad_corrected` plutôt que `sd`, dont la valeur est
  dominée par l'activation massive en position 0.
