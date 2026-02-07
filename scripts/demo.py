#!/usr/bin/env python
"""Thin vertical slice demo: demonstrates full nanocortex pipeline.

This script runs through the complete flow:
1. PDF ingestion with images
2. Multimodal retrieval query
3. Policy rule triggering human-in-the-loop
4. Approve/reject workflow
5. Learning feedback submission
6. Audit trail dump
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nanocortex.api.orchestrator import NanoCortex
from nanocortex.models.domain import PolicyRule, PolicyVerdict


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main() -> None:
    print_section("🧠 nanocortex: Thin Vertical Slice Demo")

    # Check if sample PDF exists
    sample_pdf = Path("data/sample/renewable_energy_report.pdf")
    if not sample_pdf.exists():
        print("⚠️  Sample PDF not found. Generating...")
        from scripts.generate_sample_pdf import generate_sample_pdf
        generate_sample_pdf(sample_pdf)
        print(f"✅ Generated: {sample_pdf}")

    # Initialize system
    print_section("1️⃣ System Initialization")
    cortex = NanoCortex()
    print("✅ NanoCortex initialized")
    print(f"   - Audit dir: {cortex._settings.audit_dir}")
    print(f"   - Data dir: {cortex._settings.data_dir}")
    print(f"   - Policy rules: {len(cortex.policy_engine.rules)}")

    # Add a custom policy for demo
    cortex.policy_engine.add_rule(PolicyRule(
        name="renewable_energy_expert",
        description="Require approval for energy cost queries",
        condition="contains:cost|price|lcoe",
        verdict=PolicyVerdict.NEEDS_APPROVAL,
    ))
    print(f"   - Added custom policy: renewable_energy_expert")

    # Ingest document
    print_section("2️⃣ Document Ingestion (Perception Layer)")
    result = await cortex.ingest(sample_pdf)
    print(f"✅ Document ingested: {result['filename']}")
    print(f"   - Pages: {result['pages']}")
    print(f"   - Text blocks: {result['text_blocks']}")
    print(f"   - Images: {result['images']}")
    print(f"   - Chunks indexed: {result['chunks_indexed']}")

    # Query 1: Simple retrieval (should allow)
    print_section("3️⃣ Query: General Information (Knowledge + Reasoning)")
    decision1 = await cortex.query(
        "What is the global installed capacity of solar PV?",
        top_k=3,
        strategy="hybrid",
    )
    print(f"📋 Query: {decision1['query']}")
    print(f"🤖 Answer: {decision1['answer']}")
    print(f"   - State: {decision1['state']}")
    print(f"   - Evidence chunks: {decision1['evidence_count']}")
    print(f"   - Model: {decision1['model_used']}")
    if decision1['evidence']:
        print(f"   - Top evidence (score={decision1['evidence'][0]['score']:.3f}):")
        print(f"     \"{decision1['evidence'][0]['text'][:100]}...\"")

    # Submit feedback for learning
    fb1 = cortex.submit_feedback(
        decision_id=decision1['decision_id'],
        rating="correct",
        explanation="Answer correctly states 1,500 GW from document",
    )
    print(f"✅ Feedback recorded: {fb1['rating']}")

    # Query 2: Cost query (should trigger human-in-the-loop)
    print_section("4️⃣ Query: Cost Information (Triggers Policy)")
    decision2 = await cortex.query(
        "What is the LCOE cost comparison for solar vs wind?",
        top_k=3,
    )
    print(f"📋 Query: {decision2['query']}")
    print(f"🤖 Answer: {decision2['answer']}")
    print(f"   - State: {decision2['state']}")

    print("\n📜 Policy evaluations:")
    for ev in decision2['policy_evaluations']:
        status = "🔴 MATCHED" if ev['matched'] else "⚪ not matched"
        print(f"   - {ev['rule']}: {status} → {ev['verdict']}")

    # Human-in-the-loop simulation
    if decision2['state'] == 'waiting_approval':
        print_section("5️⃣ Human-in-the-Loop: Approval Required")
        print(f"⏳ Decision {decision2['decision_id'][:8]}... requires approval")
        print("   Simulating human approval...")

        approval = cortex.approve_decision(decision2['decision_id'])
        print(f"✅ Approved! Final state: {approval['state']}")

    # Submit feedback
    fb2 = cortex.submit_feedback(
        decision_id=decision2['decision_id'],
        rating="correct",
    )
    print(f"✅ Feedback recorded: {fb2['rating']}")

    # Learning stats
    print_section("6️⃣ Learning Loop Statistics")
    stats = cortex.get_learning_stats()
    print(f"📊 Accuracy: {stats['accuracy']['accuracy']:.1%}")
    print(f"   - Total feedback: {stats['feedback_count']}")
    print(f"   - Adjustments made: {stats['adjustment_count']}")
    print(f"   - Breakdown: {stats['accuracy']['breakdown']}")

    # Audit trail
    print_section("7️⃣ Audit Trail (Sample)")
    events = cortex.get_audit_trail()
    print(f"📋 Total events logged: {len(events)}")
    print("\n   Recent events:")
    for event in events[-5:]:
        print(f"   [{event['layer']}] {event['event_type']}")

    # Dump full trace for first decision
    print(f"\n📋 Decision trace for {decision1['decision_id'][:8]}...:")
    trace = cortex.get_audit_trail(decision1['decision_id'])
    for event in trace:
        print(f"   {event['timestamp'][:19]} | {event['layer']:12} | {event['event_type']}")

    print_section("✅ Demo Complete")
    print("The thin vertical slice demonstrated:")
    print("  ✓ PDF ingestion with images")
    print("  ✓ Hybrid retrieval (BM25 + vector)")
    print("  ✓ Multi-model reasoning (orchestrator + auditor)")
    print("  ✓ Policy enforcement → human-in-the-loop")
    print("  ✓ Feedback recording → learning loop")
    print("  ✓ Full audit trail")
    print(f"\n📁 Audit logs saved to: {cortex._settings.audit_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
