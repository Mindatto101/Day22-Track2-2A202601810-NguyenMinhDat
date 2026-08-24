# Day 22 Lab Evidence

This directory contains the submission evidence for the LangSmith, prompt
versioning, RAGAS, and Guardrails tasks.

## Prompt comparison

- V1 is deliberately concise (2-4 sentences) to reduce unsupported details.
- V2 uses a structured expert style (3-5 sentences) to improve completeness.
- The final comparison must consider faithfulness together with answer relevancy:
  a longer answer is only better when its additional claims remain grounded in
  the retrieved context.

## Evidence checklist

- `01_langsmith_traces.png`: pending manual UI capture. API verification found
  97 successful `rag-query` roots (at least 50 required).
- `02_prompt_hub.png`: pending manual UI capture. Both prompts were pushed and
  pulled successfully:
  - [nguyen-minh-dat-day22-rag-v1](https://smith.langchain.com/prompts/nguyen-minh-dat-day22-rag-v1/93d2da79)
  - [nguyen-minh-dat-day22-rag-v2](https://smith.langchain.com/prompts/nguyen-minh-dat-day22-rag-v2/748501a1)
- `02_ab_routing_log.txt`: complete; 50 successful traces, V1=19 and V2=31.
- `03_ragas_scores.png`: capture the V1/V2 score table printed by Step 3.
- `03_ragas_report.json`: generated automatically by Step 3.
- `04_pii_demo_log.txt`: complete; six cases covering clean text and four PII types.
- `04_json_demo_log.txt`: complete; five cases covering valid JSON, three repairs,
  and the safe fallback.

## Verified status

- Task 1: complete — FAISS/LCEL pipeline and more than 50 successful root traces.
- Task 2: complete — two Hub prompts, deterministic MD5 routing, 50/50 successful
  A/B traces.
- Task 3: generation checkpoint complete (50 V1 + 50 V2); metric evaluation pending.
- Task 4: complete — PII redaction and JSON repair demos pass.

Replace this general comparison with the measured V1/V2 scores after Step 3
has completed, then state which prompt won and why.
