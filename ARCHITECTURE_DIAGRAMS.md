# Bluestift Cognitive Kernel — Diagrams

Visual schemas of the Kernel. Mermaid renders natively on GitHub and in most deck
tools (Notion, Obsidian, mermaid.live → export PNG/SVG for slides).

---

## 1. System architecture

```mermaid
flowchart TB
    RAYA["RAYA (Next.js)<br/>tuteur conversationnel"] -->|HTTP /analyze| API

    subgraph KERNEL["COGNITIVE KERNEL — FastAPI"]
      API["main.py · routes"]
      subgraph CORE["core/"]
        BKT["bkt · K + maîtrise"]
        FOR["forgetting · oubli"]
        MIND["mindset · M"]
        DET["detector · root-cause"]
        CAL["calibration · params vivants"]
        ANO["anomaly · sécurité"]
        GR["graph · Kernel Graph"]
      end
      subgraph SERV["services/"]
        LLM["llm · Groq → Gemini"]
        KCR["kc_registry · KCs dynamiques"]
        ANALYZE["analyze · pipeline"]
        GB["graph_builder · cold-start"]
        DBS["db · accès Supabase"]
      end
    end

    API --> CORE
    API --> SERV
    SERV --> DB[("Supabase / PostgreSQL<br/>graphe · états K,V,P,M · logs · alertes")]
    LLM --> EXT["Groq gpt-oss-120b<br/>Gemini 3.1-flash-lite"]
```

---

## 2. The /analyze pipeline (step by step)

```mermaid
flowchart TD
    IN["Conversation brute"] --> S1
    S1["1 · Extraction LLM<br/>concepts + tentatives + signaux affectifs"] --> S2
    S2["2 · KCs dynamiques<br/>get_or_create_kc (toute matière)"] --> S3
    S3["3 · Oubli<br/>K_effectif = K · e^(−λ·jours)"] --> S4
    S4["4 · Vecteur cognitif K,V,P,M<br/>BKT + gate signal fort"] --> S5
    S5["5 · Détection anomalies<br/>faux mastery, dépendance passive…"] --> S6
    S6["6 · Root-cause DFS<br/>par convergence"] --> S7
    S7["7 · Explication LLM"] --> OUT

    OUT["Diagnostic"] --> R1["root_gap + detection_path"]
    OUT --> R2["mastery_map K,V,P,M"]
    OUT --> R3["recommended_path"]
    OUT --> R4["alerts (sécurité)"]
```

---

## 3. The cognitive vector (per student × per KC)

```mermaid
flowchart LR
    KC["Knowledge Component<br/>(ex: fonction_affine)"]
    KC --> K["K · maîtrise<br/>BKT p(L)"]
    KC --> V["V · vitesse<br/>p(T) individualisé"]
    KC --> P["P · persistance<br/>1 − slip, modulé par M"]
    M["M · mindset<br/>(global à l'élève)"] -. module .-> P
```

---

## 4. Root-cause detection by convergence

Deux concepts échoués (`derivation_fonction`, `equations_lineaires`) **convergent**
sur une même lacune fondamentale → c'est la racine, même si ce n'est pas la chaîne
la plus longue. (Arête = prérequis → concept.)

```mermaid
flowchart BT
    VAR["reconnaitre_variables<br/>★ ROOT GAP"]:::root --> EXPR["ecrire_expressions"]:::unknown
    EXPR --> FA["fonction_affine"]:::unknown
    FA --> FAP["fonctions_affines_pentes"]:::unknown
    FAP --> TV["tableau_variations"]:::unknown
    TV --> DF["derivation_fonction"]:::fail
    VAR --> EL["equations_lineaires"]:::fail

    classDef root fill:#ff6b6b,color:#fff,stroke:#c92a2a,stroke-width:2px
    classDef fail fill:#ffd93d,stroke:#f08c00
    classDef unknown fill:#e9ecef,stroke:#adb5bd
```

- 🟡 **fail** = l'élève échoue (preuve directe)
- ⬜ **unknown** = jamais pratiqué, traversé par le DFS (suspect)
- 🔴 **root** = lacune racine retenue (convergence + preuve + profondeur)

---

## 5. The flywheel — static priors → dynamic, self-calibrating

```mermaid
flowchart LR
    subgraph START["Jour 1 · Cold start (statique)"]
      PRIORS["Priors littérature<br/>Corbett, Yudelson…"]
      SEED["Graphe distillé<br/>des LLMs"]
    end

    START --> COLLECT["RAYA collecte<br/>conversations réelles"]
    COLLECT --> ANALYZE["Kernel analyse"]
    ANALYZE --> STORE[("DB stocke<br/>interactions + états")]
    STORE --> CALIB["Auto-calibration<br/>λ perso, slip, difficulté KC"]
    CALIB --> SHARP["Diagnostic plus précis<br/>pour NOTRE population"]
    SHARP --> COLLECT

    CALIB -. data-gated .-> RDKT["Post-MVP :<br/>Responsible-DKT<br/>neural-symbolique"]
```

---

## 6. School → AI → Student channel (post-MVP differentiator)

```mermaid
flowchart TB
    SCHOOL["École / Enseignant"] -->|curriculum, priorités, objectifs, règles| LAYERS[("school_curriculum_layers")]
    LAYERS --> KERNEL["Kernel<br/>(pondère séquencement,<br/>contraint le graphe)"]
    LAYERS --> RAYAP["RAYA<br/>(prompt layers 3/4)"]
    KERNEL --> STUDENT["Élève"]
    RAYAP --> STUDENT
    STUDENT -->|signal| KERNEL
    KERNEL -->|alertes auditables| DASH["Dashboard institutionnel<br/>K,V,P,M · risques · équité"]
    DASH --> SCHOOL
```
