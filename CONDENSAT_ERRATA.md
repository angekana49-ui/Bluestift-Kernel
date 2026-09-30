# Errata du condensat stratégique (corpus Bluestift)

*Audit du 2026-09-30. Ce document ne remplace pas le condensat : il dit, section
par section, ce qui est faux, ce qui est à vérifier, et ce que le Kernel fait
réellement. Les valeurs des constantes et leur provenance sont dans
[PARAMETERS.md](PARAMETERS.md).*

Légende : ❌ faux · ⚠️ à nuancer ou à vérifier · ✅ correct · 🔧 état dans le code

---

## Principe

Le condensat mélange trois choses qu'il faut garder séparées :

1. **ce que dit la littérature**, avec sa source ;
2. **nos hypothèses de conception**, légitimes mais à présenter comme telles ;
3. **des transpositions** d'un résultat vers une règle de calcul, que la source
   ne dit pas.

Les erreurs sérieuses sont presque toutes dans la catégorie 3. Toute constante
est un point de départ, remplacé à terme par nos données (EM sur
`kernel.learning_events`).

---

## §1.1 Architecture — Responsible-DKT

- ❌ **Contradiction interne.** « DKT comme backbone » (§1.1) et « le Kernel ne
  nécessite pas de données d'entraînement » (Lacune 6) ne peuvent pas être vrais
  ensemble : un DKT est un réseau entraîné.
- ⚠️ **« DKT +25 % d'AUC sur BKT »** (Piech 2015) : le gain venait en grande partie
  de doublons dans ASSISTments (Xiong et al. 2016). Un BKT étendu égale DKT
  (Khajah, Lindsey & Mozer 2016).
- ⚠️ **« AUC 0.90, 10 % des données »** (Hooshyar 2026) : à vérifier dans le
  papier avant toute citation.
- 🔧 **Le Kernel est un BKT étendu + des règles symboliques.** Il n'y a aucun
  réseau de neurones. C'est défendable, et c'est surtout vrai.

## §1.2 Vecteur (K, V, P, M)

- ✅ K : seuil de maîtrise 0.95 (Corbett & Anderson 1995).
- ✅ V = p(T) individualisé (Yudelson, Koedinger & Gordon 2013).
- ⚠️ **P = 1 − slip est notre construction.** Le slip est une erreur d'inattention
  alors que le concept est su, pas de la persistance. 🔧 V et P sont stockés mais
  n'entrent dans aucune inférence.
- ⚠️ **M mélange deux construits** : le mindset de Dweck (des croyances) et la
  métacognition. « Luckin éléments 3-7 » désigne 5 éléments, alors que 4 sont
  listés. 🔧 M est estimé par le LLM à partir du texte de la conversation ; il ne
  mesure ni le temps passé ni les choix de difficulté.
- ❌ **« ΔK pondéré par la difficulté du KC »** : en BKT, la difficulté *est déjà*
  dans p(T), p(G) et p(S). Un multiplicateur en plus la compterait deux fois.
- ⚠️ **Crédit partiel {0, 0.3, 0.6, 0.7, 0.8, 1.0}** (Ostrow & Heffernan) : à
  vérifier. 🔧 Le Kernel accepte tout crédit dans [0, 1] comme preuve souple,
  sans arrondir aux paliers : l'arrondi transformait le neutre 0.5 en preuve de
  maîtrise.
- ⚠️ **« Google Effect »** : le résultat original est de Sparrow, Liu & Wegner
  (2011) ; Marsh & Rajaram en font une revue.

## §1.3 Cold start vs warm

- ⚠️ Baker 2021 (AUC 0.49 / 0.65, écart de 0.02 au 3e essai) : à vérifier.
- ❌ **Le « 3e essai » est devenu à tort un seuil de mise à jour** (Tension 3). Chez
  Baker, c'est le moment où les données propres à l'élève deviennent plus
  informatives que le prior. Ce n'est pas le moment où l'on commence à mettre à
  jour.

## §1.4 Mise à jour

- ❌ **« Les erreurs pèsent plus (gradient Hooshyar) »** transpose une observation
  sur l'entraînement d'un réseau en règle de mise à jour. BKT est **déjà**
  asymétrique : avec slip 0.1 et guess 0.2, en partant de K = 0.5, un échec fait
  −0.39 et une réussite +0.32. Ajouter une règle compterait deux fois.
- ❌ **« Seuil de mise à jour sélective » (SkillEvolBench)** : mettre à jour l'état
  à chaque observation ne crée pas de dérive. La dérive vient de la
  re-estimation des *paramètres* sur trop peu de données. 🔧 Le gate a été
  supprimé : il gelait un élève qui réussissait une fois par session, et sa
  maîtrise décroissait pendant qu'il continuait de réussir. La prudence est
  désormais là où elle sert : dans les seuils de calibration.
- ⚠️ **« Assisté ≈ autonome − 10 % (Corbett & Anderson, Étude 3) »** : à vérifier.
  De mémoire, l'étude porte sur la sur-estimation de la performance à l'examen
  par le tutoriel. 🔧 Une tentative assistée est une observation à part, avec un
  guess de 0.5 : une réussite aidée compte, mais beaucoup moins. Le plafond à 0.9
  lui laissait ~70 % du poids d'une réussite autonome.
- ✅ Taux d'inconsistance > 0.40 sur 20 interactions : implémenté. Il mesure la
  stabilité de l'*estimation*, pas un comportement de l'élève.

## §1.5 Patterns anormaux

- ❌ **« Faux mastery = exactement la dépendance passive de Bastani »** (§1.8) :
  faux. La dépendance passive *gonfle les réussites* (l'élève réussit parce qu'il
  est aidé, ce qui ressemble à du guess). Elle ne produit pas de slip élevé. Les
  « non-ideal rules » de Corbett relèvent d'un problème de transfert. Ce sont
  trois phénomènes distincts.
- ⚠️ **Surcharge cognitive** : 🔧 aujourd'hui « taux d'échec ≥ 0.5 + blocage
  conceptuel ». Cela se déclenche sur tout élève en difficulté, pas
  spécifiquement sur la surcharge.

## §1.6 Granularité

- ✅ Les courbes d'apprentissage doivent décroître. Le principe vient plutôt de
  Koedinger / DataShop. 🔧 Ce n'est pas implémenté : c'est un bon indicateur de
  qualité des KC à ajouter.
- ⚠️ **« MDP de séquencement »** : 🔧 non implémenté. `recommended_path` remonte le
  chemin de détection, puis suit les priorités de l'école.

## §1.7 OOD

- ⚠️ **« Tous les modèles de KT sont nord-américains »** : exagéré. EdNet (Corée),
  Junyi (Taïwan) et Eedi (Royaume-Uni) existent. Dis « majoritairement ».
- ⚠️ Chez Amodei, la loi de Goodhart relève du reward hacking, pas du
  distributional shift.
- 🔧 La recalibration locale est implémentée : EM MAP par KC, avec rétrécissement
  vers les priors de la littérature.

## §1.8 Critère de maîtrise

- ✅ La condition duale est la bonne. 🔧 Elle est appliquée depuis l'audit (avant,
  K ≥ 0.7 suffisait), y compris « sur les 3 dernières tentatives autonomes ».

## §2 RAYA

- ✅ **Bastani 2024** : les chiffres sont justes. ⚠️ Mais GPT Tutor (avec
  garde-fous) **n'a pas amélioré l'examen**, il a seulement supprimé la perte.
  Les garde-fous évitent le mal ; ils ne prouvent pas un gain.
- ⚠️ **« Un échec de récupération améliore l'apprentissage suivant »** : c'est
  Kornell, Hays & Bjork (2009) et Richland et al. (2009), pas Roediger & Karpicke.
- ⚠️ **Séquence EMT d'AutoTutor** : pump → hint → **prompt** → assertion ; l'étape
  « prompt » manque.
- ⚠️ **Dweck** : c'est l'éloge de la capacité (« tu es intelligent ») qui pose
  problème (Mueller & Dweck 1998), pas « Bravo » en soi. Les effets moyens des
  interventions mindset sont faibles (Sisk et al. 2018, d ≈ 0.08). La menace du
  stéréotype a des problèmes de réplication, et sa transposition au contexte
  subsaharien est une hypothèse.
- ⚠️ **Bloom 2-sigma** : le chiffre est correct, mais il vient de petites études.
  Les méta-analyses donnent +0.4 à +0.8 σ (VanLehn 2011 ; Nickow et al. 2020).
  C'est une cible, jamais une promesse.
- ⚠️ LPITutor (FA 0.94 / 0.88) : à vérifier.

## §3 Content Graph

- ⚠️ **GraphRAG (Edge et al. 2024)** désigne des résumés de communautés sur un
  graphe extrait par LLM. « Quels prérequis l'élève n'a pas maîtrisés » est un
  parcours de graphe. 🔧 `/prerequisite_gaps` fait ce parcours, ce qui est la
  bonne méthode, mais citer Edge ici est abusif.
- ⚠️ GenMentor (recall 0.48) : à vérifier.
- ⚠️ **« BKT couvre le procédural, les autres types demandent d'autres modèles »** :
  🔧 le Kernel applique BKT à tous les types ; seul λ varie selon le type.

## §4 Institutionnel

- ⚠️ DEC 2024 : de mémoire, « ne répond pas pleinement aux attentes », ce qui n'est
  pas la même chose qu'« insatisfaits ». Vérifier la formulation exacte.
- ⚠️ « 56 % vs 24 % » (Corbett & Anderson, Étude 4) et « 200 M d'élèves » : à
  sourcer.

## §5 Tensions

- **Tension 2** ⚠️ : « dès que N élèves font la même erreur → garde-fou
  automatique » contredit ton propre principe de validation humaine par
  l'enseignant.
- **Tension 3** ❌ : voir §1.4, le gate est supprimé. 🔧 La concurrence est
  réglée « comme une blockchain » : file FIFO par élève (premier arrivé,
  premier servi) et état versionné (chaque écriture est le bloc suivant ;
  une écriture calculée sur un état périmé est rejetée et recalculée).
  `learning_events` est le registre.
- **Tension 4** ❌ **La formule de M est fausse, dans deux sens.**
  - Le signe de l'abandon est inversé : un élève qui abandonne plus obtient un M
    plus élevé.
  - Avec des entrées dans [0, 1], la sigmoïde donne [0.50, 0.73]. « Fixed »
    (≤ 0.4) est donc impossible.
  - « Même logique que BKT » est faux : BKT n'utilise pas de sigmoïde.
  - « Dégradation rapide, reconstruction lente » est une hypothèse, pas du Dweck.
  - 🔧 Corrigé dans le code : sigmoïde centrée, gain 6, moyenne mobile
    symétrique 0.3.
  - 🔧 **M est désormais mesuré.**
    - « Abandon post-erreur » est mesuré dans la conversation (un échec est-il
      suivi d'un nouvel essai ?).
    - Ta formule est combinée à la dynamique d'apprentissage mesurée : V, la
      progression de K entre sessions, la récupération après erreur, P sans M
      (1 − slip, pour éviter la circularité, puisque P dépend de M).
    - La part mesurée croît avec les données et plafonne à 50 %.
  - ⚠️ **Choix assumé : ni K ni l'oubli n'entrent dans M.** Ils mesurent le niveau
    et la mémoire, pas la croyance. Les y mettre étiquetterait « fixed » un élève
    faible ou qui oublie vite, à cause de son niveau : c'est exactement le
    mécanisme de menace du stéréotype (§2.7).

## §6 Lacunes

- **Lacune 2 (oubli)** ⚠️ : K × e^(−λΔt) tend vers 0 alors que K est une
  probabilité. L'élève qui a su puis oublié passait sous celui qui n'a jamais vu
  le concept. 🔧 Le Kernel décroît désormais vers p_init. Les λ (demi-vies de 69 /
  35 / 14 jours) sont nos choix. 🔧 **L'oubli dépend des jours ET des
  révisions** : chaque récupération autonome réussie après ≥ 1 jour multiplie la
  demi-vie par √2, un échec (lapse) l'annule. C'est la forme par comptage de la
  half-life regression (Settles & Meeder 2016), à ajuster sur `learning_events`.
- **Lacune 3 (τ)** ✅ 🔧 Implémenté, neutre à 0.5 (toutes les règles y retombent
  sur l'ancien comportement). τ module :
  - ce que vaut une réponse partielle (crédit^(2τ)), c'est la « vitesse de mise
    à jour » ;
  - le seuil de maîtrise (0.93 → 0.98) ;
  - le prior de λ ;
  - et il est exposé à RAYA pour le point d'entrée EMT.

  Il ne touche jamais un paramètre ajusté sur les données (p_transit reste
  calibré par EM).
- **Lacune 4 (langue)** : 🔧 implémenté depuis l'audit. Un échec dû à un blocage
  linguistique ne compte pas contre K ; un échec ambigu compte pour moitié.
  ⚠️ « Anglais = langue pivot » : les labels et les prompts sont en français.
- **Lacune 6** ❌ : les paramètres ne viennent pas d'ASSISTments ou de Khan (ce
  sont des priors génériques), et « le LLM a DKT et Ebbinghaus dans ses poids »
  n'est pas une base de conception : connaître une théorie n'est pas
  l'appliquer.

## §8 Chaîne causale

- ⚠️ « Sweller explique pourquoi la résolution conventionnelle *n'apprend pas* » :
  elle apprend *moins bien les schémas*.
- ⚠️ Remplacer « Responsible-DKT » par « BKT étendu + règles symboliques » tant que
  le passage à un modèle hybride n'est pas justifié par nos données.

---

## Ce qui est mesuré aujourd'hui

`scripts/eval_kernel.py` (élèves synthétiques à lacune connue, 3 sessions,
extraction parfaite), comparé au code d'avant l'audit :

- précision : 15–22 % → 36–41 % ;
- fausses alertes sur élèves sans lacune : ~100 % → 8–17 % ;
- rappel : 20–29 % → 27–40 %.

**Le levier suivant, mesuré** : quand la lacune a été pratiquée, elle est trouvée
dans 30–44 % des cas ; quand elle ne l'a jamais été, dans 0–11 %. Faire sonder par
RAYA le prérequis non vérifié est l'amélioration la plus rentable.
