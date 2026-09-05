# AUC Semantic Search - Review Sign-off

Short sign-off packet for MLS and MLE review. **Source of truth for design content stays in Appendix docs** (repo `docs/*.md`). This page only tracks review scope, reviewer status, follow-up questions, and author resolutions.

---

## 1. How to use

1. Review the linked MLS or MLE page.
2. Answer the predefined questions with short notes.
3. Add follow-up questions in the page-level question log.
4. Author records resolution or links the decision/ticket.
5. Both reviewers update the roll-up below.

**Confluence tree (manual nest after upload):**

```
AUC Semantic Search - Review Sign-off
├── MLS Technical Review
├── MLE Service Review
├── Decisions And Open Questions
└── Appendix → link existing Discovery / Plan / Design / Architecture / Process / Playbook / Infrastructure pages
```

---

## 2. Review scope

| Field | Value |
| ----- | ----- |
| Review round | 1 |
| Review date | |
| Repo | [https://github.com/gdcorp-dna/auc-semantic-search](https://github.com/gdcorp-dna/auc-semantic-search) |
| Docs path | `auc-semantic-search/docs/` |
| Scope | Phase 1 filters-only / QI + Phase 2 design intent |

---

## 3. Stakeholder pages

| Role | Page | Owns | Status |
| ---- | ---- | ---- | ------ |
| MLS | [MLS Technical Review](REVIEW_MLS.md) | QI quality, filter grounding, eval, hybrid-first ML intent | Not started |
| MLE | [MLE Service Review](REVIEW_MLE.md) | Service shape, API/contracts, degrade, deploy/ops | Not started |
| - | [Decisions And Open Questions](DECISIONS.md) | ADRs + parking lot from review comments | Open |

Add Product / Security / Platform pages later using the same pattern.

---

## 4. Reviewer sign-off

| Role | Reviewer | Date | Status | Blocking? | Notes |
| ---- | -------- | ---- | ------ | --------- | ----- |
| MLS | | | | Yes / No | |
| MLE | | | | Yes / No | |

**Status values:** Approved · Approved with follow-ups · Changes requested · Not reviewed

**Gate:** both MLS and MLE Approved or Approved with follow-ups before treating this packet as signed for Phase 1 build continuity.

---

## 5. Cross-review follow-ups

Use this log when a question or resolution should be visible to both reviewers.

| ID | Reviewer | Question / Concern | Author resolution | Decision / Ticket link | Status |
| -- | -------- | ------------------ | ----------------- | ---------------------- | ------ |
| Q1 | | | | | Open |

**Status values:** Open · Answered · Deferred · Closed

---

## 6. Appendix

Link Confluence pages backed by repo files. Do not copy bodies into this packet.

| Doc | Repo file |
| --- | --------- |
| Discovery Document | `DISCOVERY.md` |
| Delivery Plan | `PLAN.md` |
| Design Document | `DESIGN.md` |
| Architecture | `ARCHITECTURE.md` |
| Process Flow | `PROCESS.md` |
| Playbook | `PLAYBOOK.md` |
| Observability (CW alarms / filters) | `OBSERVABILITY.md` |
| Infrastructure and Deployment | `INFRASTRUCTURE.md` |
