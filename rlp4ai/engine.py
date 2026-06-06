from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import (
    CrossRef, Edge, Message, Node, NodeStatus, NodeType,
    RelType, RoleType, SemanticGraph, SpeechAct, SpeechActSet,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        status = "OK" if self.valid else "FAIL"
        lines = [f"ValidationResult({status})"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


AMBIGUITY_THRESHOLD = 0.15
QUALITY_RISK_THRESHOLD = 0.5

@dataclass
class WeightResolution:
    node_id: int
    rule: int 
    blocked: bool
    quality_effective: float
    requires_query: bool
    requires_risk_node: bool
    description: str


def resolve_concurrent_edges(
    graph: SemanticGraph,
    target_node_id: int,
) -> Optional[WeightResolution]:

    blocking_edges = [
        e for e in graph.edges_to(target_node_id)
        if e.rel_type.is_blocking
    ]
    enabling_edges = [
        e for e in graph.edges_to(target_node_id)
        if e.rel_type.is_enabling
    ]

    if not blocking_edges:
        return None

    w_block = max((e.weight for e in blocking_edges), default=0.0)
    w_enable = max((e.weight for e in enabling_edges), default=0.0)
    diff = abs(w_block - w_enable)

    if diff < AMBIGUITY_THRESHOLD:
        return WeightResolution(
            node_id=target_node_id,
            rule=3,
            blocked=False,
            quality_effective=0.0,
            requires_query=True,
            requires_risk_node=False,
            description=(
                f"AMBIGUOUS: |{w_block:.2f} - {w_enable:.2f}| = {diff:.2f} < {AMBIGUITY_THRESHOLD}. "
                f"Obbligatorio QUERY prima di procedere."
            ),
        )
    elif w_block > w_enable:
        return WeightResolution(
            node_id=target_node_id,
            rule=1,
            blocked=True,
            quality_effective=0.0,
            requires_query=False,
            requires_risk_node=False,
            description=(
                f"BLOCKED: weight_BLOCKS({w_block:.2f}) > weight_ENABLES({w_enable:.2f}). "
                f"Obbligatorio WARN o DENY."
            ),
        )
    else:
        q_eff = w_enable * (1.0 - w_block)
        needs_risk = q_eff < QUALITY_RISK_THRESHOLD
        return WeightResolution(
            node_id=target_node_id,
            rule=2,
            blocked=False,
            quality_effective=q_eff,
            requires_query=False,
            requires_risk_node=needs_risk,
            description=(
                f"ENABLED_DEGRADED: quality_effective = {w_enable:.2f} * (1 - {w_block:.2f}) = {q_eff:.2f}."
                + (f" RISK/QUALITY_DEGRADED obbligatorio (q_eff < {QUALITY_RISK_THRESHOLD})."
                   if needs_risk else "")
            ),
        )


@dataclass
class ConstraintCheck:
    constraint_node_id: int
    satisfied: bool
    quality_found: float
    quality_required: float
    description: str


def check_constraint_quality(
    graph: SemanticGraph,
    constraint_node_id: int,
) -> ConstraintCheck:
    constraint = graph.get_node(constraint_node_id)
    if constraint is None or constraint.type != NodeType.CONSTRAINT:
        return ConstraintCheck(
            constraint_node_id, True, 1.0, 1.0,
            "Nodo non trovato o non di tipo CONSTRAINT."
        )

    threshold = constraint.quality_threshold
    requires_edges = [
        e for e in graph.edges_from(constraint_node_id)
        if e.rel_type == RelType.REQUIRES
    ]

    min_quality = 1.0
    for req_edge in requires_edges:
        target_id = req_edge.to_id
        enabling = [
            e for e in graph.edges_to(target_id)
            if e.rel_type.is_enabling
        ]
        if enabling:
            q = min(e.quality for e in enabling)
            min_quality = min(min_quality, q)

    satisfied = min_quality >= threshold
    return ConstraintCheck(
        constraint_node_id=constraint_node_id,
        satisfied=satisfied,
        quality_found=min_quality,
        quality_required=threshold,
        description=(
            f"CONSTRAINT '{constraint.value}': "
            f"quality_found={min_quality:.2f} "
            f"{'≥' if satisfied else '<'} "
            f"quality_threshold={threshold:.2f} → "
            f"{'SATISFIED' if satisfied else 'VIOLATED — obbligatorio WARN o OVERRIDE_REQUEST'}."
        ),
    )


def check_condition_ref(
    edge: Edge,
    local_graph: SemanticGraph,
    session_store: Optional["SessionStore"] = None,
) -> tuple[bool, str]:
    ref = edge.condition_ref
    if ref is None:
        return True, "Nessun condition_ref — arco sempre attivo."

    if CrossRef.is_cross_ref(ref):
        if session_store is None:
            return False, f"condition_ref cross-turno '{ref}' ma nessun SessionStore disponibile."
        try:
            cr = CrossRef.parse(ref)
        except ValueError as exc:
            return False, str(exc)
        node = session_store.get_node(cr)
        if node is None:
            return False, f"Riferimento '{ref}' non risolvibile — obbligatorio QUERY (§5.5.3)."
        active = node.is_positive()
        return active, (
            f"condition_ref '{ref}' → nodo status={node.status.value} → "
            f"{'ACTIVE' if active else 'INACTIVE'}."
        )
    else:
        try:
            local_id = int(ref)
        except (ValueError, TypeError):
            return False, f"condition_ref locale '{ref}' non è un intero valido."
        node = local_graph.get_node(local_id)
        if node is None:
            return False, f"condition_ref locale node_id={local_id} non trovato nel grafo corrente."
        active = node.is_positive()
        return active, (
            f"condition_ref locale node_id={local_id} → status={node.status.value} → "
            f"{'ACTIVE' if active else 'INACTIVE'}."
        )


def should_accept_broadcast(
    msg: Message,
    receiver_session_id: str,
    receiver_role: RoleType,
) -> tuple[bool, bool, str]:
    rb = msg.role_binding
    scope = rb.broadcast_scope()

    if scope is None:
        return (rb.target_id == receiver_session_id or
                rb.target_id.startswith(receiver_session_id),
                False, "Messaggio diretto.")

    if scope == "SYSTEM":
        return True, False, "BROADCAST@SYSTEM — tutti gli agenti."

    if scope == receiver_session_id:
        return True, False, f"BROADCAST@{scope} — sessione corrente."

    if scope == rb.session_id and scope != receiver_session_id:
        priority_ok = rb.priority >= 0.95
        cross_refs_with_high_weight = _has_cross_session_ref(
            msg.semantic_graph, receiver_session_id, min_weight=0.90
        )
        if priority_ok and cross_refs_with_high_weight:
            return True, True, (
                f"EXCEPTION_ACCEPT: BROADCAST da {rb.session_id} con "
                f"priority={rb.priority} e riferimenti cross-sessione a "
                f"{receiver_session_id} → accettato per regola §5.5.5."
            )
        return False, False, (
            f"BROADCAST da sessione {rb.session_id} — scope formale = {scope}, "
            f"receiver in {receiver_session_id} — non incluso. "
            f"(priority={rb.priority}, cross_refs={cross_refs_with_high_weight})"
        )

    return False, False, f"BROADCAST scope={scope} non include {receiver_session_id}."


def _has_cross_session_ref(
    graph: SemanticGraph,
    target_session: str,
    min_weight: float = 0.90,
) -> bool:
    for edge in graph.edges:
        for ref_str in (str(edge.from_id), str(edge.to_id), edge.condition_ref or ""):
            if CrossRef.is_cross_ref(ref_str):
                try:
                    cr = CrossRef.parse(ref_str)
                    if cr.session_id == target_session and edge.weight >= min_weight:
                        return True
                except ValueError:
                    pass
    return False



class MessageValidator:
    def __init__(self, session_store: Optional["SessionStore"] = None):
        self.session_store = session_store

    def validate(self, msg: Message) -> ValidationResult:
        result = ValidationResult(valid=True)
        self._check_ttl(msg, result)
        self._check_layer1(msg, result)
        self._check_layer2(msg, result)
        self._check_layer3(msg, result)
        self._check_concurrent_weights(msg, result)
        self._check_constraint_quality(msg, result)
        self._check_condition_refs(msg, result)
        self._check_speech_act_consistency(msg, result)
        return result

    def _check_ttl(self, msg: Message, r: ValidationResult) -> None:
        if msg.role_binding.is_expired():
            r.add_error(
                f"Messaggio scaduto: TTL={msg.role_binding.ttl_ms}ms, "
                f"timestamp={msg.role_binding.timestamp_us}."
            )

    def _check_layer1(self, msg: Message, r: ValidationResult) -> None:
        rb = msg.role_binding
        if rb.turn_idx < 0:
            r.add_error("turn_idx deve essere >= 0.")
        if not (0.0 <= rb.priority <= 1.0):
            r.add_error(f"priority={rb.priority} fuori range [0,1].")
        if not rb.sender_id:
            r.add_error("sender_id mancante.")
        if not rb.session_id:
            r.add_error("session_id mancante.")

    def _check_layer2(self, msg: Message, r: ValidationResult) -> None:
        st = msg.state_tensor
        if not (0.0 <= st.confidence <= 1.0):
            r.add_error(f"confidence={st.confidence} fuori range [0,1].")

    def _check_layer3(self, msg: Message, r: ValidationResult) -> None:
        g = msg.semantic_graph
        ids = [n.node_id for n in g.nodes]
        if len(ids) != len(set(ids)):
            r.add_error("node_id duplicati nel grafo.")
        for e in g.edges:
            if e.rel_type in (RelType.CONDITIONAL_ENABLES, RelType.CONDITIONAL_BLOCKS):
                if e.condition_ref is None:
                    r.add_error(
                        f"Arco {e.rel_type.value} da {e.from_id}→{e.to_id} "
                        f"deve avere condition_ref (§5.5.2)."
                    )
        for e in g.edges:
            if not (0.0 <= e.weight <= 1.0):
                r.add_error(f"Arco {e.from_id}→{e.to_id}: weight={e.weight} fuori range.")
            if not (0.0 <= e.quality <= 1.0):
                r.add_error(f"Arco {e.from_id}→{e.to_id}: quality={e.quality} fuori range.")


    def _check_concurrent_weights(self, msg: Message, r: ValidationResult) -> None:
        g = msg.semantic_graph
        target_ids = {e.to_id for e in g.edges if isinstance(e.to_id, int)}
        for tid in target_ids:
            res = resolve_concurrent_edges(g, tid)
            if res is None:
                continue
            if res.blocked:
                r.add_warning(f"Nodo {tid}: {res.description}")
            elif res.requires_query:
                r.add_warning(f"Nodo {tid}: {res.description}")
            elif res.requires_risk_node:
                has_risk = any(
                    n.type in (NodeType.RISK, NodeType.QUALITY_DEGRADED)
                    for n in g.nodes
                )
                if not has_risk:
                    r.add_error(
                        f"Nodo {tid}: quality_effective={res.quality_effective:.2f} < "
                        f"{QUALITY_RISK_THRESHOLD} — obbligatorio nodo RISK o "
                        f"QUALITY_DEGRADED nel grafo (§5.5.1 Regola 2)."
                    )

    def _check_constraint_quality(self, msg: Message, r: ValidationResult) -> None:
        g = msg.semantic_graph
        for node in g.nodes:
            if node.type != NodeType.CONSTRAINT:
                continue
            chk = check_constraint_quality(g, node.node_id)
            if not chk.satisfied:
                r.add_warning(f"Nodo {node.node_id}: {chk.description}")

    def _check_condition_refs(self, msg: Message, r: ValidationResult) -> None:
        g = msg.semantic_graph
        for e in g.edges:
            if e.condition_ref is None:
                continue
            active, desc = check_condition_ref(e, g, self.session_store)
            if not active:
                r.add_warning(f"Arco {e.from_id}→{e.to_id}: condition_ref inattivo — {desc}")

    def _check_speech_act_consistency(self, msg: Message, r: ValidationResult) -> None:
        sa = msg.semantic_graph.speech_act
        if SpeechAct.WARN in sa.acts and len(sa.acts) == 1:
            has_blocked = any(
                n.status == NodeStatus.BLOCKED for n in msg.semantic_graph.nodes
            )
            if has_blocked:
                r.add_warning(
                    "speech_act=WARN con nodi BLOCKED nel grafo: "
                    "considerare WARN+DENY per chiarire il blocco (§5.5.4)."
                )
