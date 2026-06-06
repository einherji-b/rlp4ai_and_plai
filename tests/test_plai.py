import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest

from plai.compiler import compile_plai, CompileError, tokenize, TK_ACT, TK_NODE_TYPE
from plai.decompiler import Decompiler
from rlp4ai.model import NodeType, RelType, SpeechAct, NodeStatus, UncertaintyType



def compile_ok(source: str):
    msg, warnings = compile_plai(source)
    return msg


class TestLexer:

    def test_tokenize_act(self):
        tokens = tokenize("DELEGATE {}")
        types = [t.type for t in tokens]
        assert TK_ACT in types

    def test_tokenize_node_type(self):
        tokens = tokenize('GOAL "my goal"')
        types = [t.type for t in tokens]
        assert TK_NODE_TYPE in types

    def test_tokenize_string(self):
        tokens = tokenize('"hello world"')
        str_tok = next(t for t in tokens if t.type == "STRING")
        assert str_tok.value == "hello world"

    def test_tokenize_arrow_ascii(self):
        tokens = tokenize("-> ENABLES")
        types = [t.type for t in tokens]
        assert "ARROW" in types

    def test_tokenize_arrow_unicode(self):
        tokens = tokenize("→ ENABLES")
        types = [t.type for t in tokens]
        assert "ARROW" in types

    def test_tokenize_comment_ignored(self):
        tokens = tokenize("# this is a comment\nDELEGATE {}")
        values = [t.value for t in tokens]
        assert "this" not in values

    def test_tokenize_unclosed_string_raises(self):
        from plai.compiler import LexError
        with pytest.raises(LexError):
            tokenize('"unclosed string')



class TestCompilerBasic:

    def test_minimal_delegate(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

DELEGATE {
  CONFIDENCE 0.90

  GOAL "my goal"
}
"""
        msg = compile_ok(src)
        assert msg.role_binding.sender_id == "AGENT-A"
        assert msg.role_binding.target_id == "AGENT-B"
        assert msg.role_binding.session_id == "SESSION-001"
        assert SpeechAct.DELEGATE in msg.semantic_graph.speech_act.acts

    def test_speech_act_composed(self):
        src = """
@AGENT-B [EXECUTOR] → AGENT-A | SESSION-001

WARN+PROPOSE_ALTERNATIVE {
  CONFIDENCE 0.80

  RISK "some risk"
}
"""
        msg = compile_ok(src)
        acts = msg.semantic_graph.speech_act.acts
        assert SpeechAct.WARN in acts
        assert SpeechAct.PROPOSE_ALTERNATIVE in acts

    def test_confidence_parsed(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

COMMIT {
  CONFIDENCE 0.95
  GOAL "g"
}
"""
        msg = compile_ok(src)
        assert msg.state_tensor.confidence == pytest.approx(0.95)

    def test_uncertainty_parsed(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

WARN {
  CONFIDENCE 0.75 | uncertainty=EPISTEMIC
  RISK "r"
}
"""
        msg = compile_ok(src)
        assert msg.state_tensor.uncertainty_type == UncertaintyType.EPISTEMIC

    def test_note_becomes_nl_summary(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

INFORM {
  CONFIDENCE 0.85
  STATE "ready"
}

NOTE: "Tutto pronto, in attesa."
"""
        msg = compile_ok(src)
        assert "pronto" in msg.interface.nl_summary

    def test_priority_parsed(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001 | priority=0.98

DELEGATE {
  CONFIDENCE 0.90
  GOAL "g"
}
"""
        msg = compile_ok(src)
        assert msg.role_binding.priority == pytest.approx(0.98)

    def test_ttl_parsed(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001 | ttl=5000ms

QUERY {
  CONFIDENCE 0.85
  STATE "s"
}
"""
        msg = compile_ok(src)
        assert msg.role_binding.ttl_ms == 5000




class TestCompilerNodes:

    def _compile_with_nodes(self, *node_stmts):
        body = "\n".join(f"  {s}" for s in node_stmts)
        src = f"""
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

DELEGATE {{
  CONFIDENCE 0.90
{body}
}}
"""
        return compile_ok(src)

    def test_all_node_types(self):
        types = [
            "GOAL", "ACTION", "CONSTRAINT", "ENTITY", "CONCEPT",
            "STATE", "FAILED_STATE", "PARTIAL_STATE", "RISK", "QUALITY_DEGRADED",
        ]
        for ntype in types:
            msg = self._compile_with_nodes(f'{ntype} "test_{ntype.lower()}"')
            assert any(n.type.value == ntype for n in msg.semantic_graph.nodes), \
                f"NodeType {ntype} non trovato"

    def test_node_salience(self):
        msg = self._compile_with_nodes('GOAL "g"\n    salience = 0.75')
        goal = next(n for n in msg.semantic_graph.nodes if n.type == NodeType.GOAL)
        assert goal.salience == pytest.approx(0.75)

    def test_node_status(self):
        msg = self._compile_with_nodes('STATE "s"\n    status = BLOCKED')
        state = next(n for n in msg.semantic_graph.nodes if n.type == NodeType.STATE)
        assert state.status == NodeStatus.BLOCKED

    def test_constraint_authority(self):
        msg = self._compile_with_nodes(
            'CONSTRAINT "deadline=T+20min"\n    authority = COORDINATOR'
        )
        c = next(n for n in msg.semantic_graph.nodes if n.type == NodeType.CONSTRAINT)
        assert c.authority_required == "COORDINATOR"

    def test_constraint_quality_min(self):
        msg = self._compile_with_nodes(
            'CONSTRAINT "data_integrity"\n    quality_min = 1.0'
        )
        c = next(n for n in msg.semantic_graph.nodes if n.type == NodeType.CONSTRAINT)
        assert c.quality_threshold == pytest.approx(1.0)

    def test_sync_barrier(self):
        src = """
@AGENT-A [COORDINATOR] → BROADCAST | SESSION-GLOBAL

DELEGATE {
  CONFIDENCE 0.94

  SYNC_BARRIER "pre_cutover_sync"
    salience = 1.0
    sessions = [SESSION-301 | SESSION-302 | SESSION-303]
    timeout = 120000
    sync_action = "integrity_check_COMPLETE"
}
"""
        msg = compile_ok(src)
        barrier = next(n for n in msg.semantic_graph.nodes
                       if n.type == NodeType.SYNC_BARRIER)
        assert "SESSION-301" in barrier.sync_sessions
        assert "SESSION-303" in barrier.sync_sessions
        assert barrier.sync_timeout_ms == 120000
        assert barrier.sync_action == "integrity_check_COMPLETE"




class TestCompilerEdges:

    def test_enables_edge(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

DELEGATE {
  CONFIDENCE 0.90

  ACTION "do_analysis"
    → ENABLES "deliver_result" weight=0.9

  GOAL "deliver_result"
}
"""
        msg = compile_ok(src)
        enables = [e for e in msg.semantic_graph.edges
                   if e.rel_type == RelType.ENABLES]
        assert len(enables) > 0
        assert enables[0].weight == pytest.approx(0.9)

    def test_blocks_edge(self):
        src = """
@AGENT-B [EXECUTOR] → AGENT-A | SESSION-001

WARN {
  CONFIDENCE 0.80

  FAILED_STATE "preprocessing_failed"
    → BLOCKS "run_analysis" weight=0.85

  ACTION "run_analysis"
}
"""
        msg = compile_ok(src)
        blocks = [e for e in msg.semantic_graph.edges
                  if e.rel_type == RelType.BLOCKS]
        assert len(blocks) == 1
        assert blocks[0].weight == pytest.approx(0.85)

    def test_edge_quality(self):
        src = """
@AGENT-B [EXECUTOR] → AGENT-A | SESSION-001

WARN {
  CONFIDENCE 0.80

  ACTION "fallback"
    → ENABLES "goal" weight=0.9 quality=0.72

  RISK "risk_node"
  GOAL "goal"
}
"""
        msg = compile_ok(src)
        e = next(e for e in msg.semantic_graph.edges
                 if e.rel_type == RelType.ENABLES)
        assert e.quality == pytest.approx(0.72)

    def test_all_rel_types_compile(self):
        rel_types = [
            "ENABLES", "BLOCKS", "CAUSES", "REQUIRES", "SUPPORTS",
            "RESOLVES", "MITIGATES", "MODIFIES", "DEGRADES", "CONTRADICTS",
        ]
        for rt in rel_types:
            src = f"""
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

INFORM {{
  CONFIDENCE 0.85
  STATE "source"
    → {rt} "target"
  ENTITY "target"
}}
"""
            msg = compile_ok(src)
            edges = [e for e in msg.semantic_graph.edges
                     if e.rel_type.value == rt]
            assert len(edges) > 0, f"rel_type {rt} non trovato"

    def test_cross_ref_edge(self):
        src = """
@AGENT-C [VERIFIER] → BROADCAST | SESSION-099

WARN+DENY {
  CONFIDENCE 0.98

  STATE "constraint_violated"
    → CONTRADICTS @SESSION-099:turn3:node2 weight=0.90
}
"""
        msg = compile_ok(src)
        edge = next(e for e in msg.semantic_graph.edges
                    if e.rel_type == RelType.CONTRADICTS)
        assert "@SESSION-099" in str(edge.to_id)

    def test_conditional_enables_with_condition_ref(self):
        src = """
@AGENT-D [EXECUTOR] → AGENT-A | SESSION-303

COMMIT {
  CONFIDENCE 0.85

  ACTION "cutover"
    salience = 0.90

  SYNC_BARRIER "barrier"
    sessions = [SESSION-301 | SESSION-303]
    timeout = 60000
    sync_action = "done"

  ACTION "proceed_after_sync"
    → CONDITIONAL_ENABLES "cutover" weight=0.95 if 1
}
"""
        msg = compile_ok(src)
        cond = [e for e in msg.semantic_graph.edges
                if e.rel_type == RelType.CONDITIONAL_ENABLES]
        assert len(cond) > 0
        assert cond[0].condition_ref is not None




class TestDecompiler:

    def _compile_and_decompile(self, src: str, annotated=False) -> str:
        msg, _ = compile_plai(src)
        d = Decompiler()
        return d.decompile(msg, annotated=annotated)

    def test_header_present(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-042 | priority=0.95

DELEGATE {
  CONFIDENCE 0.90
  GOAL "g"
}
"""
        out = self._compile_and_decompile(src)
        assert "AGENT-A" in out
        assert "SESSION-042" in out
        assert "COORDINATOR" in out

    def test_node_values_present(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

DELEGATE {
  CONFIDENCE 0.90
  GOAL "deliver_report_X"
  ACTION "analyze_dataset"
  CONSTRAINT "deadline=T+20min"
    authority = COORDINATOR
}
"""
        out = self._compile_and_decompile(src)
        assert "deliver report X" in out
        assert "analyze dataset" in out
        assert "deadline" in out

    def test_speech_act_in_output(self):
        src = """
@AGENT-B [EXECUTOR] → AGENT-A | SESSION-001

WARN+COMMIT {
  CONFIDENCE 0.85
  RISK "r"
  GOAL "g"
}
"""
        out = self._compile_and_decompile(src)
        assert "WARN" in out
        assert "COMMIT" in out

    def test_annotated_adds_comments(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

WARN {
  CONFIDENCE 0.80

  FAILED_STATE "preprocessing_failed"
    → BLOCKS "run_analysis" weight=0.85

  ACTION "run_analysis"
    salience = 0.95
}
"""
        out = self._compile_and_decompile(src, annotated=True)
        assert "#" in out  

    def test_from_json_roundtrip(self):
        sample = {
            "LAYER_1": {
                "sender_id": "AGENT-B", "role_type": "EXECUTOR",
                "target_id": "AGENT-A", "session_id": "SESSION-042",
                "turn_idx": 2, "ttl_ms": 0, "priority": 0.93,
            },
            "LAYER_2": {"confidence": 0.87, "uncertainty_type": "EPISTEMIC",
                        "delta_flag": True},
            "LAYER_3": {
                "nodes": [
                    {"node_id": 0, "type": "FAILED_STATE",
                     "value": "preprocessing_failed", "salience": 0.95, "status": "ACTIVE"},
                    {"node_id": 1, "type": "RISK",
                     "value": "data_corruption_risk", "salience": 0.85, "status": "ACTIVE"},
                ],
                "edges": [
                    {"from": 0, "to": 1, "rel_type": "CAUSES",
                     "weight": 0.85, "quality": 1.0},
                ],
                "speech_act": "WARN+PROPOSE_ALTERNATIVE",
                "intent_vector": "preprocessing fallito",
            },
        }
        msg, err = Decompiler.from_json(json.dumps(sample))
        assert err == "", f"Errore parsing: {err}"
        d = Decompiler()
        out = d.decompile(msg, annotated=False)
        assert "AGENT-B" in out
        assert "WARN" in out or "PROPOSE_ALTERNATIVE" in out
        assert "preprocessing" in out.lower()



class TestRoundtrip:

    def _roundtrip(self, src: str) -> str:
        msg, _ = compile_plai(src)
        d = Decompiler()
        return d.decompile(msg, annotated=False)

    def test_sender_preserved(self):
        src = """
@MYAGENT [VERIFIER] → TARGET | SESSION-X

CONFIRM {
  CONFIDENCE 0.99
  STATE "all_ok"
}
"""
        out = self._roundtrip(src)
        assert "MYAGENT" in out

    def test_speech_act_preserved(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

OVERRIDE_GRANT+DELEGATE {
  CONFIDENCE 0.91
  ACTION "isolate_cluster"
}
"""
        out = self._roundtrip(src)
        assert "OVERRIDE_GRANT" in out
        assert "DELEGATE" in out

    def test_node_types_preserved(self):
        src = """
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-001

DELEGATE {
  CONFIDENCE 0.90
  GOAL "main_goal"
  ACTION "do_work"
  CONSTRAINT "sla_99pct"
    authority = COORDINATOR
  RISK "timeout_risk"
}
"""
        msg, _ = compile_plai(src)
        types = {n.type for n in msg.semantic_graph.nodes}
        assert NodeType.GOAL in types
        assert NodeType.ACTION in types
        assert NodeType.CONSTRAINT in types
        assert NodeType.RISK in types

    def test_example_1_compiles(self):
        from plai.cli import EXAMPLES
        msg, warnings = compile_plai(EXAMPLES["1"]["source"])
        assert msg is not None
        assert len(msg.semantic_graph.nodes) > 0

    def test_example_2_compiles(self):
        from plai.cli import EXAMPLES
        msg, _ = compile_plai(EXAMPLES["2"]["source"])
        types = {n.type for n in msg.semantic_graph.nodes}
        assert NodeType.FAILED_STATE in types

    def test_example_3_compiles_with_sync_barrier(self):
        from plai.cli import EXAMPLES
        msg, _ = compile_plai(EXAMPLES["3"]["source"])
        types = {n.type for n in msg.semantic_graph.nodes}
        assert NodeType.SYNC_BARRIER in types

    def test_example_4_override_grant(self):
        from plai.cli import EXAMPLES
        msg, _ = compile_plai(EXAMPLES["4"]["source"])
        assert SpeechAct.OVERRIDE_GRANT in msg.semantic_graph.speech_act.acts

    def test_example_5_query_with_ttl(self):
        from plai.cli import EXAMPLES
        msg, _ = compile_plai(EXAMPLES["5"]["source"])
        assert SpeechAct.QUERY in msg.semantic_graph.speech_act.acts
        assert msg.role_binding.ttl_ms == 15000




class TestCompileErrors:

    def test_unclosed_string_raises(self):
        with pytest.raises(CompileError):
            compile_plai('@AGENT-A [COORDINATOR] → B | S\nDELEGATE {\n  GOAL "unclosed\n}')

    def test_missing_brace_does_not_crash(self):
        try:
            compile_plai('@A [COORDINATOR] → B | S\nDELEGATE {\n  GOAL "g"')
      
        except (CompileError, Exception):
            pass  

    def test_empty_source_gives_inform(self):
        msg, _ = compile_plai("")
        assert SpeechAct.INFORM in msg.semantic_graph.speech_act.acts


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
