# nanocortex - Unified AI System

<div align="center">

![nanocortex](https://img.shields.io/badge/nanocortex-Unified%20AI%20System-22c55e?style=for-the-badge&logo=brain&logoColor=white)

**Perceive • Reason • Act • Learn • Audit**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.23+-009688?style=flat-square&logo=adobe-acrobat-reader&logoColor=white)](https://pymupdf.readthedocs.io)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-E92063?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![pytest](https://img.shields.io/badge/pytest-7.0+-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org)

</div>

---

## 🎯 Overview

**nanocortex** is a production-grade, modular AI decision and control platform demonstrating how multimodal AI systems can perceive, reason, act, and learn under explicit constraints—with full auditability and human control.

- 🔍 **Perceives** documents via PDF/image ingestion with OCR
- 📚 **Retrieves** citation-grounded evidence without hallucinations
- 🧠 **Reasons** with policy enforcement and approval workflows
- 📈 **Learns** from outcomes without retraining base models
- 📋 **Audits** every decision from input to outcome

> ⚠️ **This is not a chatbot.** This is a *decision machine* capable of acting under constraints, explaining its outputs, and improving behavior over time.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    nanocortex Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Layer 1: Perception & Extraction            │   │
│  │        PDF Ingestion • OCR • Bounding-Box Grounding      │   │
│  │     (derived from: dex, gradia, imgshape)                │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Layer 2: Knowledge & Retrieval              │   │
│  │      Hybrid RAG (BM25 + Vector) • Citation Tracking      │   │
│  │     (derived from: iai-solutions-task, agentic-rag)      │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Layer 3: Reasoning & Control                │   │
│  │    Stateful Agents • Policy Engine • Human-in-the-Loop   │   │
│  │     (derived from: CloudRedux, antigravity)              │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Layer 4: Adaptation & Learning              │   │
│  │   Feedback Loop • Mistake Tracking • Auto-Adjustments    │   │
│  │     (derived from: Huemn.AI)                             │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Layer 5: Observability & Audit              │   │
│  │    Decision Traces • Evidence References • Override Logs │   │
│  │     (cross-cutting across all layers)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Derived From

This repository unifies and subsumes prior evaluated projects into a cohesive reference architecture:

| Layer | Derived Repositories | Contribution |
|-------|---------------------|--------------|
| **Perception** | [dex](https://github.com/STiFLeR7/dex), [gradia](https://github.com/STiFLeR7/gradia), [imgshape](https://github.com/STiFLeR7/imgshape) | Document ingestion, extraction pipelines, image diagnostics |
| **Knowledge** | [iai-solutions-task](https://github.com/STiFLeR7/iai-solutions-task), [agentic-rag](https://github.com/STiFLeR7/agentic-rag) | Citation-grounded RAG, evidence handling |
| **Reasoning** | [CloudRedux](https://github.com/STiFLeR7/CloudRedux), [antigravity](https://github.com/STiFLeR7/antigravity) | Policy-first agent control, stateful decisions |
| **Learning** | [Huemn.AI](https://github.com/STiFLeR7/Huemn.AI) | Post-run evaluation, learning loop |

---

## 🩺 Real-World Testing: Clinical Decision Support

nanocortex was tested on a **Dermatology AI Decision Support** use case with 10 medical research PDFs:

```
📁 Ingested: 4 PDFs (1,365 chunks indexed)
   - A Comprehensive Review of the Acne.pdf (9 pages, 415 chunks)
   - A global perspective on the epidemiology of acne.pdf (21 pages, 337 chunks)
   - Artificial_Intelligence_in_the_Assessment_and_Grad.pdf (13 pages, 435 chunks)
   - assessment_of_life_quality_index_among_patients.pdf (6 pages, 178 chunks)
```

### Clinical Policies Applied

| Policy | Condition | Verdict |
|--------|-----------|---------|
| `treatment_approval` | contains: treatment, prescription | NEEDS_APPROVAL |
| `severity_grading` | contains: severe, moderate, grade | NEEDS_APPROVAL |
| `sensitive_populations` | contains: pediatric, pregnancy | NEEDS_APPROVAL |

### Query Results

| Query | State | Policy Triggered |
|-------|-------|-----------------|
| "What is the global prevalence of acne in adolescents?" | ✅ completed | — |
| "How effective is AI in grading acne severity?" | ✅ completed | — |
| "What treatment options are recommended for moderate acne?" | ⏸️ waiting_approval | `treatment_approval`, `severity_grading` |
| "How does acne affect quality of life in patients?" | ✅ completed | — |

### Demo Output Highlights

```
📋 Query: What treatment options are recommended for moderate acne?
   State: waiting_approval

   🔐 Policy Evaluations:
      - treatment_approval: 🔴 MATCHED → needs_approval
      - severity_grading: 🔴 MATCHED → needs_approval

   ⏳ Decision 4a0f7895... requires clinician approval
   🔵 Simulating clinician approval...
   ✅ Approved! Final state: completed
```

### Results

| Metric | Value |
|--------|-------|
| Accuracy | **100%** |
| Feedback recorded | 4 |
| Audit events | 26 |
| Human approvals | 1 |

👉 **See [examples/](./examples/)** for the full demo

---

## 📂 Project Structure

```
nanocortex/
├── src/nanocortex/
│   ├── api/
│   │   └── orchestrator.py      # NanoCortex entry point
│   ├── perception/
│   │   └── ingestion.py         # PDF/image extraction + OCR
│   ├── knowledge/
│   │   └── retriever.py         # Hybrid BM25 + vector RAG
│   ├── reasoning/
│   │   ├── agent.py             # Multi-model decision agent
│   │   └── policy.py            # Externalized rule engine
│   ├── learning/
│   │   └── feedback.py          # Outcome tracking + adjustments
│   ├── audit/
│   │   └── logger.py            # JSON-Lines event log
│   ├── models/
│   │   └── domain.py            # Pydantic domain models
│   └── config.py                # Environment configuration
│
├── data/
│   ├── audit/                   # Audit logs (JSON-Lines)
│   └── sample/                  # Sample documents
│
├── scripts/
│   ├── demo.py                  # Thin vertical slice demo
│   └── generate_sample_pdf.py   # PDF generator
│
└── tests/                       # pytest suite (27 tests)
```

---

## 📡 REST API (Service Mode)

Run nanocortex as a microservice:

```powershell
uvicorn nanocortex.api.server:app --reload --port 8000
```

Open **<http://localhost:8000/docs>** for interactive Swagger docs.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check + stats |
| `/v1/ingest` | POST | Upload and ingest document |
| `/v1/ingest/path` | POST | Ingest from local file path |
| `/v1/query` | POST | Query with retrieval + reasoning |
| `/v1/decisions/{id}/approve` | POST | Approve pending decision |
| `/v1/decisions/{id}/reject` | POST | Reject with reason |
| `/v1/feedback/{id}` | POST | Submit learning feedback |
| `/v1/learning/stats` | GET | Learning loop statistics |
| `/v1/policies` | GET/POST | List or add policy rules |
| `/v1/audit` | GET | Get audit trail |
| `/v1/audit/{id}` | GET | Get decision trace |

### Example: Query via API

```bash
# Ingest a document
curl -X POST http://localhost:8000/v1/ingest/path \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/sample/renewable_energy_report.pdf"}'

# Query
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the global solar capacity?",
    "strategy": "hybrid"
  }'

# Approve decision
curl -X POST http://localhost:8000/v1/decisions/abc123/approve
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Tesseract OCR (optional, for scanned PDFs)

### 1. Clone & Setup Environment

```powershell
# Clone the repository
git clone https://github.com/STiFLeR7/nanocortex.git
cd nanocortex

# Create Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# LLM Providers (Optional - fallback mode works without these)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Paths
AUDIT_DIR=./data/audit
DATA_DIR=./data
```

### 3. Run the Demo

```powershell
# Generate sample PDF
python scripts/generate_sample_pdf.py

# Run thin vertical slice demo
python scripts/demo.py
```

### 4. Run Tests

```powershell
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 💻 Usage

```python
import asyncio
from nanocortex.api.orchestrator import NanoCortex

async def main():
    # Initialize the system
    cortex = NanoCortex()
    
    # 1. Ingest a document
    result = await cortex.ingest("data/sample/renewable_energy_report.pdf")
    print(f"Indexed {result['chunks_indexed']} chunks")
    
    # 2. Query with retrieval + reasoning
    decision = await cortex.query("What is the global solar capacity?")
    print(f"Answer: {decision['answer']}")
    print(f"State: {decision['state']}")
    
    # 3. Handle human-in-the-loop if required
    if decision['state'] == 'waiting_approval':
        cortex.approve_decision(decision['decision_id'])
    
    # 4. Submit feedback for learning
    cortex.submit_feedback(
        decision_id=decision['decision_id'],
        rating="correct",
    )
    
    # 5. View audit trail
    events = cortex.get_audit_trail(decision['decision_id'])
    print(f"Audit events: {len(events)}")

asyncio.run(main())
```

---

## 🔐 Policy Engine

Policies are **data, not code**—they can be loaded from config files:

```python
from nanocortex.models.domain import PolicyRule, PolicyVerdict

# Require approval for sensitive queries
cortex.policy_engine.add_rule(PolicyRule(
    name="sensitive_topics",
    condition="contains:financial|medical|legal",
    verdict=PolicyVerdict.NEEDS_APPROVAL,
))

# Deny answers with no evidence
cortex.policy_engine.add_rule(PolicyRule(
    name="no_hallucination",
    condition="no_evidence",
    verdict=PolicyVerdict.DENY,
))
```

---

## 📊 Learning Loop

The system learns from outcomes **without retraining base models**:

| Feature | Description |
|---------|-------------|
| **Feedback Recording** | Correct, incorrect, hallucination ratings |
| **Pattern Detection** | Automatic threshold-based triggers |
| **Behavioral Adjustments** | Retrieval weights, prompt patches |
| **Persistence** | State saved across runs |

```python
stats = cortex.get_learning_stats()
print(f"Accuracy: {stats['accuracy']['accuracy']:.1%}")
print(f"Adjustments made: {stats['adjustment_count']}")
```

---

## 🛠️ Development Status

| Component | Status |
|-----------|--------|
| Perception Layer | ✅ Complete |
| Knowledge Layer | ✅ Complete |
| Reasoning Layer | ✅ Complete |
| Learning Layer | ✅ Complete |
| Audit Layer | ✅ Complete |
| Test Suite | ✅ 27 tests passing |
| Demo Script | ✅ Complete |

---

## 🔜 Recommended Stack

| Role | Model | Purpose |
|------|-------|---------|
| **Orchestrator** | GPT-5.2 Codex | Generates answers from evidence |
| **Auditor** | Claude Opus 4.6 | Reviews for hallucinations |
| **Ingestion Helper** | KimiK 2.5 | Document digestion (optional) |

> One model **acts**, one model **judges**, one model **reads**. No single-model arrogance.

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by STIFLER**

*A reference architecture demonstrating how AI should be allowed to act.*

</div>
