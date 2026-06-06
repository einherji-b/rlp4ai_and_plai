import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from rlp4ai.model import (
    CrossRef, Edge, InterfaceLayer, Message, Modality, Node, NodeStatus,
    NodeType, RelType, RoleBinding, RoleType, SemanticGraph, SpeechAct,
    SpeechActSet, StateTensor, UncertaintyType, Compression,
)
from rlp4ai.engine import (
    MessageValidator, WeightResolution,
    check_constraint_quality, check_condition_ref,
    resolve_concurrent_edges, should_accept_broadcast,
    AMBIGUITY_THRESHOLD,
)
from rlp4ai.session import SessionStore, SyncBarrierState




def make_rb(
    sender="AGENT-A", role=RoleType.COORDINATOR,
    target="AGENT-B", session="SESSION-001", turn=1,
    ttl=0, priority=0.9,
) -> RoleBinding:
    return RoleBinding(
        sender_id=sender, role_type=role,
        target_id=target, session_id=session,
        turn_idx=turn, ttl_ms=ttl, priority=priority,
    )

def make_st(confidence=0.9, uncertainty=UncertaintyType.NONE) -> StateTensor:
    return StateTensor(confidence=confidence, uncertainty_type=uncertainty)

def make_msg(rb=None, st=None, sg=None) -> Message:
    return Message(
        role_binding=rb or make_rb(),
        state_tensor=st or make_st(),
        semantic_graph=sg or SemanticGraph(),
    )



class TestSpeechActSet:

    def test_parse_single(self):
        sa = SpeechActSet.parse("WARN")
        assert SpeechAct.WARN in sa
        assert not sa.blocks()
        assert sa.warns()

    def test_parse_composed(self):
        sa = SpeechActSet.parse("WARN+DENY")
        assert sa.warns()
        assert sa.blocks()

    def test_warn_does_not_block(self):
        sa = SpeechActSet.parse("WARN")
        assert not sa.blocks(), "WARN da solo non deve bloccare (§5.5.4)"

    def test_warn_deny_blocks(self):
        sa = SpeechActSet.parse("WARN+DENY")
        assert sa.blocks()

    def test_warn_commit(self):
        sa = SpeechActSet.parse("WARN+COMMIT")
        assert sa.warns() and sa.commits() and not sa.blocks()

    def test_str_roundtrip(self):
        sa = SpeechActSet.parse("DENY+WARN")
        assert "WARN" in str(sa)
        assert "DENY" in str(sa)

    def test_all_speech_acts_parseable(self):
        for act in SpeechAct:
            sa = SpeechActSet.parse(act.value)
            assert act in sa


class TestCrossRef:

    def test_parse_valid(self):
        cr = CrossRef.parse("@SESSION-099:turn2:node3")
        assert cr.session_id == "SESSION-099"
        assert cr.turn_idx == 2
        assert cr.node_id == 3

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            CrossRef.parse("SESSION-099:turn2:node3") 

    def test_is_cross_ref(self):
        assert CrossRef.is_cross_ref("@SESSION-001:turn1:node0")
        assert not CrossRef.is_cross_ref("3")
        assert not CrossRef.is_cross_ref("local_ref")

    def test_str(self):
        cr = CrossRef("SESSION-X", 5, 2)
        assert str(cr) == "@SESSION-X:turn5:node2"


class TestNodeStatus:

    def test_is_positive_active(self):
        n = Node(0, NodeType.GOAL, "goal_A")
        assert n.is_positive()

    def test_is_positive_blocked(self):
        n = Node(0, NodeType.GOAL, "goal_A", status=NodeStatus.BLOCKED)
        assert not n.is_positive()

    def test_is_positive_failed_state(self):
        n = Node(0, NodeType.FAILED_STATE, "something_failed")
        assert not n.is_positive()

    def test_bypassed_not_positive(self):
        n = Node(0, NodeType.CONSTRAINT, "SLA", status=NodeStatus.BYPASSED)
        assert not n.is_positive()

    def test_pending_is_positive(self):
        n = Node(0, NodeType.ACTION, "waiting", status=NodeStatus.PENDING)
        assert n.is_positive()



class TestWeightSemantics:

    def _graph_with_conflict(self, w_block, w_enable, add_risk=False):
        nodes = [Node(0, NodeType.GOAL, "target")]
        if add_risk:
            nodes.append(Node(1, NodeType.RISK, "risk_node"))
        edges = [
            Edge(from_id=10, to_id=0, rel_type=RelType.BLOCKS, weight=w_block),
            Edge(from_id=20, to_id=0, rel_type=RelType.ENABLES, weight=w_enable),
        ]
        return SemanticGraph(nodes=nodes, edges=edges,
                             speech_act=SpeechActSet.parse("INFORM"))

    def test_rule1_block_wins(self):
        g = self._graph_with_conflict(0.85, 0.60)
        res = resolve_concurrent_edges(g, 0)
        assert res is not None
        assert res.rule == 1
        assert res.blocked is True

    def test_rule2_enable_wins_degraded(self):
        g = self._graph_with_conflict(0.30, 0.80, add_risk=True)
        res = resolve_concurrent_edges(g, 0)
        assert res is not None
        assert res.rule == 2
        assert not res.blocked
        assert res.quality_effective == pytest.approx(0.80 * (1 - 0.30), abs=1e-4)

    def test_rule2_requires_risk_when_quality_low(self):

        g = self._graph_with_conflict(0.40, 0.60)
        res = resolve_concurrent_edges(g, 0)
        assert res.rule == 2
        assert res.requires_risk_node is True

    def test_rule3_ambiguous(self):
        g = self._graph_with_conflict(0.70, 0.65)
        res = resolve_concurrent_edges(g, 0)
        assert res is not None
        assert res.rule == 3
        assert res.requires_query is True

    def test_no_blocking_edges_returns_none(self):
        g = SemanticGraph(
            nodes=[Node(0, NodeType.GOAL, "g")],
            edges=[Edge(0, 0, RelType.ENABLES, 0.9)],
            speech_act=SpeechActSet.parse("INFORM"),
        )
        assert resolve_concurrent_edges(g, 0) is None

    def test_validator_errors_on_missing_risk(self):
        g = self._graph_with_conflict(0.40, 0.60, add_risk=False)
        msg = make_msg(sg=g)
        v = MessageValidator()
        result = v.validate(msg)
        assert not result.valid
        assert any("RISK" in e for e in result.errors)

    def test_validator_ok_with_risk_node(self):
        g = self._graph_with_conflict(0.45, 0.60, add_risk=True)
        msg = make_msg(sg=g)
        v = MessageValidator()
        result = v.validate(msg)
        assert result.valid


class TestQualityConstraint:

    def _graph_with_constraint(self, quality_on_edge, threshold=1.0):
        nodes = [
            Node(0, NodeType.CONSTRAINT, "data_integrity_100pct",
                 quality_threshold=threshold,
                 authority_required="MIGRATION_COMMANDER"),
            Node(1, NodeType.ACTION, "run_integrity_check"),
            Node(2, NodeType.ACTION, "auto_repair"),
        ]
        edges = [
            Edge(from_id=0, to_id=1, rel_type=RelType.REQUIRES, weight=1.0),
            Edge(from_id=2, to_id=1, rel_type=RelType.ENABLES,
                 weight=0.95, quality=quality_on_edge),
        ]
        return SemanticGraph(nodes=nodes, edges=edges,
                             speech_act=SpeechActSet.parse("INFORM"))

    def test_constraint_violated_when_quality_low(self):
        g = self._graph_with_constraint(0.90, threshold=1.0)
        chk = check_constraint_quality(g, 0)
        assert not chk.satisfied
        assert chk.quality_found == pytest.approx(0.90)
        assert chk.quality_required == pytest.approx(1.0)

    def test_constraint_satisfied_when_quality_ok(self):
        g = self._graph_with_constraint(1.0, threshold=1.0)
        chk = check_constraint_quality(g, 0)
        assert chk.satisfied

    def test_constraint_satisfied_with_lower_threshold(self):
        g = self._graph_with_constraint(0.85, threshold=0.80)
        chk = check_constraint_quality(g, 0)
        assert chk.satisfied

    def test_validator_warns_on_constraint_violation(self):
        g = self._graph_with_constraint(0.72, threshold=1.0)
        msg = make_msg(sg=g)
        result = MessageValidator().validate(msg)
        assert any("VIOLATED" in w or "quality" in w.lower()
                   for w in result.warnings)




class TestConditionalEdges:

    def test_conditional_edge_requires_condition_ref(self):
        g = SemanticGraph(
            nodes=[Node(0, NodeType.GOAL, "g"), Node(1, NodeType.ACTION, "a")],
            edges=[Edge(from_id=1, to_id=0,
                        rel_type=RelType.CONDITIONAL_ENABLES,
                        weight=0.9, condition_ref=None)],
            speech_act=SpeechActSet.parse("INFORM"),
        )
        result = MessageValidator().validate(make_msg(sg=g))
        assert not result.valid
        assert any("condition_ref" in e for e in result.errors)

    def test_condition_ref_local_active(self):
        nodes = [
            Node(0, NodeType.GOAL, "goal"),
            Node(1, NodeType.ACTION, "action"),
            Node(2, NodeType.STATE, "override_received", status=NodeStatus.ACTIVE),
        ]
        edges = [Edge(from_id=1, to_id=0,
                      rel_type=RelType.CONDITIONAL_ENABLES,
                      weight=0.9, condition_ref="2")]
        g = SemanticGraph(nodes=nodes, edges=edges,
                          speech_act=SpeechActSet.parse("INFORM"))
        active, desc = check_condition_ref(edges[0], g)
        assert active, f"Atteso ACTIVE, got: {desc}"

    def test_condition_ref_local_blocked(self):
        nodes = [
            Node(0, NodeType.GOAL, "goal"),
            Node(1, NodeType.ACTION, "action"),
            Node(2, NodeType.STATE, "condition",
                 status=NodeStatus.BLOCKED),
        ]
        edges = [Edge(from_id=1, to_id=0,
                      rel_type=RelType.CONDITIONAL_ENABLES,
                      weight=0.9, condition_ref="2")]
        g = SemanticGraph(nodes=nodes, edges=edges,
                          speech_act=SpeechActSet.parse("INFORM"))
        active, desc = check_condition_ref(edges[0], g)
        assert not active

    def test_condition_ref_cross_turn(self):
        store = SessionStore()
        override_node = Node(5, NodeType.STATE, "override_granted",
                             status=NodeStatus.ACTIVE)
        rb = make_rb(session="SESSION-099", turn=3)
        sg = SemanticGraph(nodes=[override_node],
                           speech_act=SpeechActSet.parse("OVERRIDE_GRANT"))
        store.ingest(Message(rb, make_st(), sg))

        edge = Edge(from_id=1, to_id=0,
                    rel_type=RelType.CONDITIONAL_ENABLES,
                    weight=0.9, condition_ref="@SESSION-099:turn3:node5")
        active, desc = check_condition_ref(edge,
                                           SemanticGraph(),
                                           session_store=store)
        assert active, f"Atteso ACTIVE, got: {desc}"

    def test_condition_ref_cross_turn_superseded(self):
        store = SessionStore()
        n1 = Node(3, NodeType.CONSTRAINT, "some_constraint")
        msg1 = Message(
            make_rb(session="SESSION-T", turn=1),
            make_st(),
            SemanticGraph(nodes=[n1], speech_act=SpeechActSet.parse("INFORM")),
        )
        store.ingest(msg1)
        mod_edge = Edge(from_id=10, to_id="@SESSION-T:turn1:node3",
                        rel_type=RelType.MODIFIES, weight=1.0)
        n2 = Node(10, NodeType.CONSTRAINT, "new_constraint")
        msg2 = Message(
            make_rb(session="SESSION-T", turn=2),
            make_st(),
            SemanticGraph(nodes=[n2], edges=[mod_edge],
                          speech_act=SpeechActSet.parse("INFORM")),
        )
        store.ingest(msg2)

        edge = Edge(from_id=1, to_id=0,
                    rel_type=RelType.CONDITIONAL_ENABLES,
                    condition_ref="@SESSION-T:turn1:node3",
                    weight=0.9)
        active, desc = check_condition_ref(edge, SemanticGraph(), store)
        assert not active, f"Atteso INACTIVE (SUPERSEDED), got: {desc}"



class TestSessionStore:


    def test_ingest_and_retrieve(self):
        store = SessionStore()
        node = Node(0, NodeType.CONSTRAINT, "deploy_window=04:00")
        msg = make_msg(rb=make_rb(session="S1", turn=1),
                       sg=SemanticGraph(nodes=[node],
                                        speech_act=SpeechActSet.parse("DELEGATE")))
        store.ingest(msg)
        retrieved = store.get_message("S1", 1)
        assert retrieved is not None

    def test_duplicate_turn_raises(self):
        store = SessionStore()
        msg = make_msg(rb=make_rb(session="S1", turn=1))
        store.ingest(msg)
        with pytest.raises(ValueError, match="già presente"):
            store.ingest(msg)

    def test_modifies_supersedes_old_node(self):

        store = SessionStore()
        n_orig = Node(5, NodeType.CONSTRAINT, "deploy_window=04:00")
        msg1 = Message(
            make_rb(session="S99", turn=1),
            make_st(),
            SemanticGraph(nodes=[n_orig], speech_act=SpeechActSet.parse("DELEGATE")),
        )
        store.ingest(msg1)

        n_new = Node(5, NodeType.CONSTRAINT, "deploy_window=04:30")
        mod_edge = Edge(
            from_id=5,
            to_id="@S99:turn1:node5",
            rel_type=RelType.MODIFIES,
            weight=1.0,
        )
        msg2 = Message(
            make_rb(session="S99", turn=2),
            make_st(),
            SemanticGraph(nodes=[n_new], edges=[mod_edge],
                          speech_act=SpeechActSet.parse("APPROVES")),
        )
        store.ingest(msg2)

        ref = CrossRef.parse("@S99:turn1:node5")
        old = store._get_node_by_crossref(ref)
        assert old is not None
        assert old.status == NodeStatus.SUPERSEDED

        assert store.get_node(ref) is None

    def test_cross_ref_validation_future_turn(self):
        store = SessionStore()
        valid, reason = store.validate_cross_ref("@S1:turn5:node0", current_turn=3)
        assert not valid
        assert "futuro" in reason

    def test_history_ordered(self):
        store = SessionStore()
        for t in [3, 1, 2]:
            store.ingest(make_msg(rb=make_rb(session="SX", turn=t)))
        history = store.history("SX")
        assert [m.turn_idx for m in history] == [1, 2, 3]


class TestBroadcastScope:

    def _make_broadcast_msg(self, session, priority=0.98,
                             cross_session=None, cross_weight=0.95):
        nodes = [Node(0, NodeType.ACTION, "isolate")]
        edges = []
        if cross_session:
            edge = Edge(
                from_id=0,
                to_id=f"@{cross_session}:turn1:node3",
                rel_type=RelType.ENABLES,
                weight=cross_weight,
            )
            edges.append(edge)
        rb = RoleBinding(
            sender_id="AGENT-A", role_type=RoleType.INCIDENT_COMMANDER,
            target_id="BROADCAST", session_id=session,
            turn_idx=4, priority=priority,
        )
        sg = SemanticGraph(nodes=nodes, edges=edges,
                           speech_act=SpeechActSet.parse("OVERRIDE_GRANT+DELEGATE"))
        return Message(rb, make_st(), sg)

    def test_broadcast_same_session_accepted(self):
        msg = self._make_broadcast_msg("SESSION-201")
        accept, needs_exc, _ = should_accept_broadcast(
            msg, "SESSION-201", RoleType.RESPONDER_PRIMARY
        )
        assert accept
        assert not needs_exc

    def test_broadcast_no_qualifier_different_session_rejected_low_priority(self):
        msg = self._make_broadcast_msg("SESSION-201", priority=0.80)
        accept, _, reason = should_accept_broadcast(
            msg, "SESSION-202", RoleType.RESPONDER_SECONDARY
        )
        assert not accept

    def test_broadcast_exception_accept_rule(self):
        msg = self._make_broadcast_msg(
            "SESSION-201", priority=0.98,
            cross_session="SESSION-202", cross_weight=0.95
        )
        accept, needs_exc, reason = should_accept_broadcast(
            msg, "SESSION-202", RoleType.RESPONDER_SECONDARY
        )
        assert accept, f"Atteso accettato, got: {reason}"
        assert needs_exc, "Atteso EXCEPTION_ACCEPT"

    def test_broadcast_system_accepted_everywhere(self):
        rb = RoleBinding(
            sender_id="AGENT-A", role_type=RoleType.COORDINATOR,
            target_id="BROADCAST@SYSTEM", session_id="SESSION-001",
            turn_idx=1, priority=0.9,
        )
        msg = Message(rb, make_st(), SemanticGraph(
            speech_act=SpeechActSet.parse("INFORM")))
        accept, needs_exc, _ = should_accept_broadcast(
            msg, "SESSION-999", RoleType.OBSERVER
        )
        assert accept
        assert not needs_exc



class TestSyncBarrier:

    def _make_barrier(self, sessions=None, timeout=120000) -> SyncBarrierState:
        sessions = sessions or ["SESSION-301", "SESSION-302", "SESSION-303"]
        return SyncBarrierState(
            key="pre_cutover_sync",
            sync_sessions=sessions,
            sync_timeout_ms=timeout,
            sync_action="run_integrity_check_COMPLETE",
        )

    def test_barrier_not_resolved_with_partial_commits(self):
        barrier = self._make_barrier()
        barrier.register_commit("SESSION-301")
        barrier.register_commit("SESSION-302")
        assert barrier.status == "ACTIVE"
        assert "SESSION-303" in barrier.remaining()

    def test_barrier_resolved_when_all_commit(self):
        barrier = self._make_barrier()
        barrier.register_commit("SESSION-301")
        barrier.register_commit("SESSION-302")
        resolved = barrier.register_commit("SESSION-303")
        assert resolved is True
        assert barrier.status == "RESOLVED"

    def test_barrier_blocked(self):
        barrier = self._make_barrier()
        barrier.block()
        assert barrier.status == "BLOCKED"
        barrier.register_commit("SESSION-303")
        assert barrier.status == "BLOCKED"

    def test_barrier_extension(self):
        barrier = self._make_barrier(timeout=47000)
        barrier.apply_extension(15000)
        assert barrier.effective_timeout_ms() == 62000

    def test_session_store_tracks_barrier_commits(self):
        store = SessionStore()
        barrier_node = Node(
            6, NodeType.SYNC_BARRIER, "pre_cutover_sync",
            sync_sessions=["S301", "S302", "S303"],
            sync_timeout_ms=120000,
            sync_action="integrity_check_COMPLETE",
        )
        barrier_state = store.register_barrier("pre_cutover_sync", barrier_node)

        msg_s301 = Message(
            make_rb(session="S301", turn=1),
            make_st(),
            SemanticGraph(nodes=[Node(0, NodeType.STATE, "done")],
                          speech_act=SpeechActSet.parse("COMMIT")),
        )
        store.ingest(msg_s301)
        assert "S301" in barrier_state.committed
        assert barrier_state.status == "ACTIVE"

        msg_s302 = Message(
            make_rb(session="S302", turn=1),
            make_st(),
            SemanticGraph(nodes=[Node(0, NodeType.STATE, "done")],
                          speech_act=SpeechActSet.parse("COMMIT")),
        )
        store.ingest(msg_s302)

        msg_s303 = Message(
            make_rb(session="S303", turn=2),
            make_st(),
            SemanticGraph(nodes=[Node(0, NodeType.STATE, "done")],
                          speech_act=SpeechActSet.parse("COMMIT")),
        )
        store.ingest(msg_s303)
        assert barrier_state.status == "RESOLVED"



class TestScenarioDeployFailed:

    def test_failed_state_blocks_action(self):
        nodes = [
            Node(0, NodeType.ACTION, "analyze_dataset_D"),
            Node(1, NodeType.FAILED_STATE, "preprocessing_step_P",
                 status=NodeStatus.ACTIVE),
            Node(2, NodeType.STATE, "raw_data_available"),
        ]
        edges = [
            Edge(from_id=1, to_id=0, rel_type=RelType.BLOCKS, weight=0.85),
            Edge(from_id=2, to_id=0, rel_type=RelType.ENABLES, weight=0.60),
        ]
        g = SemanticGraph(nodes=nodes, edges=edges,
                          speech_act=SpeechActSet.parse("WARN+PROPOSE_ALTERNATIVE"))
        res = resolve_concurrent_edges(g, 0)
        assert res is not None
        assert res.blocked, "FAILED_STATE (0.85) deve bloccare ACTION (enable 0.60)"
        assert res.rule == 1

    def test_response_graph_with_fallback(self):
        nodes = [
            Node(0, NodeType.ACTION, "execute_fallback_preprocessing"),
            Node(1, NodeType.ACTION, "analyze_dataset_D"),
            Node(2, NodeType.CONSTRAINT, "deadline=T+25min", quality_threshold=0.0),
            Node(3, NodeType.RISK, "high_error_rate_on_raw_data"),
            Node(4, NodeType.GOAL, "deliver_report_X"),
        ]
        edges = [
            Edge(from_id=0, to_id=1, rel_type=RelType.RESOLVES, weight=0.95, quality=1.0),
            Edge(from_id=0, to_id=1, rel_type=RelType.ENABLES, weight=0.90, quality=0.85),
            Edge(from_id=3, to_id=4, rel_type=RelType.BLOCKS, weight=0.85),
            Edge(from_id=0, to_id=3, rel_type=RelType.MITIGATES, weight=0.90),
        ]
        g = SemanticGraph(nodes=nodes, edges=edges,
                          speech_act=SpeechActSet.parse("WARN+PROPOSE_ALTERNATIVE"))
        result = MessageValidator().validate(make_msg(sg=g))
        assert result.valid, repr(result)


class TestScenarioTest44:

    def test_quality_040_violates_constraint(self):
        nodes = [
            Node(0, NodeType.CONSTRAINT, "all_3_failed_tests_must_resolve",
                 quality_threshold=1.0,
                 authority_required="COORDINATOR"),
            Node(1, NodeType.ACTION, "run_integrity_check_post_fix"),
            Node(2, NodeType.PARTIAL_STATE,
                 "tests_31_38_passed_test_44_exit_0_with_leak"),
            Node(3, NodeType.RISK, "production_memory_exhaustion"),
            Node(4, NodeType.QUALITY_DEGRADED, "service_K_runtime_stability_compromised"),
        ]
        edges = [
            Edge(from_id=0, to_id=1, rel_type=RelType.REQUIRES, weight=1.0),
            Edge(from_id=2, to_id=1, rel_type=RelType.ENABLES, weight=0.66,
                 quality=0.40), 
            Edge(from_id=2, to_id=3, rel_type=RelType.CAUSES, weight=0.95),
            Edge(from_id=3, to_id=4, rel_type=RelType.CAUSES, weight=0.85),
        ]
        g = SemanticGraph(nodes=nodes, edges=edges,
                          speech_act=SpeechActSet.parse("WARN"))
        chk = check_constraint_quality(g, 0)
        assert not chk.satisfied
        assert chk.quality_found == pytest.approx(0.40)
        assert "VIOLATED" in chk.description

    def test_warn_only_with_blocked_node_warns(self):
        nodes = [Node(0, NodeType.ACTION, "deploy", status=NodeStatus.BLOCKED)]
        g = SemanticGraph(nodes=nodes,
                          speech_act=SpeechActSet.parse("WARN"))
        result = MessageValidator().validate(make_msg(sg=g))
        assert result.valid
        assert any("WARN+DENY" in w for w in result.warnings)



class TestBypassedConstraint:

    def test_bypassed_is_not_positive(self):
        n = Node(0, NodeType.CONSTRAINT, "no_isolation_without_explicit_order",
                 status=NodeStatus.BYPASSED)
        assert not n.is_positive()

    def test_bypassed_differs_from_superseded_semantically(self):
        bypassed = Node(0, NodeType.CONSTRAINT, "SLA", status=NodeStatus.BYPASSED)
        superseded = Node(1, NodeType.CONSTRAINT, "OLD", status=NodeStatus.SUPERSEDED)
        assert not bypassed.is_positive()
        assert not superseded.is_positive()
        assert bypassed.type == NodeType.CONSTRAINT
        assert bypassed.status != superseded.status

    def test_override_grant_speech_act_parseable(self):
        sa = SpeechActSet.parse("OVERRIDE_GRANT+DELEGATE")
        assert SpeechAct.OVERRIDE_GRANT in sa
        assert SpeechAct.DELEGATE in sa


class TestTTL:

    def test_expired_message_fails_validation(self):
        rb = RoleBinding(
            sender_id="AGENT-A", role_type=RoleType.COORDINATOR,
            target_id="AGENT-B", session_id="S1", turn_idx=1,
            timestamp_us=int((time.time() - 10) * 1_000_000), 
            ttl_ms=1000,  
            priority=0.9,
        )
        msg = make_msg(rb=rb)
        result = MessageValidator().validate(msg)
        assert not result.valid
        assert any("scaduto" in e.lower() for e in result.errors)

    def test_no_ttl_never_expires(self):
        rb = RoleBinding(
            sender_id="AGENT-A", role_type=RoleType.COORDINATOR,
            target_id="AGENT-B", session_id="S1", turn_idx=1,
            timestamp_us=int((time.time() - 1000) * 1_000_000),  
            ttl_ms=0,
            priority=0.9,
        )
        assert not rb.is_expired()

    def test_valid_ttl_not_expired(self):
        rb = make_rb(ttl=60000)
        assert not rb.is_expired()


class TestChecksum:

    def test_same_message_same_checksum(self):
        nodes = [Node(0, NodeType.GOAL, "g")]
        g = SemanticGraph(nodes=nodes, speech_act=SpeechActSet.parse("INFORM"))
        rb = make_rb()
        msg = Message(rb, make_st(), g)
        assert msg.checksum() == msg.checksum()

    def test_different_nodes_different_checksum(self):
        g1 = SemanticGraph(nodes=[Node(0, NodeType.GOAL, "A")],
                           speech_act=SpeechActSet.parse("INFORM"))
        g2 = SemanticGraph(nodes=[Node(0, NodeType.GOAL, "B")],
                           speech_act=SpeechActSet.parse("INFORM"))
        m1 = Message(make_rb(), make_st(), g1)
        m2 = Message(make_rb(), make_st(), g2)
        assert m1.checksum() != m2.checksum()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
