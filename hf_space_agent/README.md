---
title: Clinical Reasoning Agent Demo
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
hardware: cpu
short_description: Agentic clinical reasoning over FHIR, with citation grounding, quarantine, and audit
tags:
  - agents
  - mcp
  - fhir
  - healthcare
  - human-in-the-loop
  - fastapi
---

Clinical decision support, never decision making. See the main repo's `docs/CLINICAL_AGENT.md` for the
full architecture, reuse map, and measured eval numbers. No live LLM is called in this Space -- see
`app.py`'s module docstring for the honest limit this demo does and doesn't prove.
