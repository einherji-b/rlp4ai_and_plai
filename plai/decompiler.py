from __future__ import annotations

import json
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlp4ai.model import (
    Edge, Message, Node, NodeStatus, NodeType, RelType,
    RoleType, SemanticGraph, SpeechAct, SpeechActSet, UncertaintyType,
)
from rlp4ai.engine import (
    MessageValidator, check_constraint_quality,
    resolve_concurrent_edges,
)


_STATUS_COMMENT = {
    NodeStatus.ACTIVE:     "",
    NodeStatus.BLOCKED:    "  # BLOCCATO",
    NodeStatus.SUPERSEDED: "  # SUPERATO",
    NodeStatus.PENDING:    "  # IN ATTESA",
    NodeStatus.RESOLVED:   "  # RISOLTO",
    NodeStatus.BYPASSED:   "  # BYPASSED",
}

_REL_COMMENT = {
    RelType.ENABLES:             "abilita",
    RelType.BLOCKS:              "BLOCCA",
    RelType.CAUSES:              "causa",
    RelType.REQUIRES:            "richiede",
    RelType.SUPPORTS:            "supporta",
    RelType.RESOLVES:            "risolve",
    RelType.MITIGATES:           "mitiga",
    RelType.MODIFIES:            "modifica",
    RelType.DEGRADES:            "degrada",
    RelType.CONTRADICTS:         "contraddice",
    RelType.IS_A:                "è_tipo_di",
    RelType.HAS_PROPERTY:        "ha_proprietà",
    RelType.TEMPORAL_BEFORE:     "prima_di",
    RelType.TEMPORAL_AFTER:      "dopo",
    RelType.PART_OF:             "parte_di",
    RelType.EQUIVALENT:          "equivale_a",
    RelType.CONDITIONAL_ENABLES: "abilita_se",
    RelType.CONDITIONAL_BLOCKS:  "BLOCCA_se",
}

_UNCERT_COMMENT = {
    UncertaintyType.NONE:      "",
    UncertaintyType.EPISTEMIC: "  # mancano dati — risolvibile",
    UncertaintyType.ALEATORIC: "  # variabilità intrinseca — non risolvibile",
    UncertaintyType.MIXED:     "  # incertezza mista",
}

_SPEECH_ACT_COMMENT = {
    SpeechAct.INFORM:                    "Informazione",
    SpeechAct.REQUEST:                   "Richiesta",
    SpeechAct.CONFIRM:                   "Conferma",
    SpeechAct.DENY:                      "Rifiuto/Blocco",
    SpeechAct.PROPOSE:                   "Proposta",
    SpeechAct.DELEGATE:                  "Delega",
    SpeechAct.QUERY:                     "Domanda — attende risposta",
    SpeechAct.ACKNOWLEDGE:               "Presa in carico",
    SpeechAct.WARN:                      "Avviso",
    SpeechAct.COMMIT:                    "Impegno formale",
    SpeechAct.PROPOSE_ALTERNATIVE:       "Proposta Alternativa",
    SpeechAct.APPROVES:                  "Approvazione",
    SpeechAct.REJECTS:                   "Rifiuto Formale",
    SpeechAct.OVERRIDE_REQUEST:          "Richiesta Override",
    SpeechAct.OVERRIDE_GRANT:            "Override Autorizzato",
    SpeechAct.TIMEOUT_ESCALATE:          "Escalation Timeout",
    SpeechAct.EXCEPTION_ACCEPT:          "Accettazione Eccezionale",
    SpeechAct.BARRIER_EXTENSION_REQUEST: "Estensione Barriera",
}


class Decompiler:

    def decompile(self, msg: Message, annotated: bool = True) -> str:
        lines: list[str] = []

        if annotated:
            lines.append(f"# RLP4AI → PLAI  |  checksum: {msg.checksum()[:12]}…")

        lines.append(self._header(msg))
        lines.append("")
        lines.append(self._act_block(msg, annotated))

        if msg.interface and msg.interface.nl_summary:
            lines.append("")
            lines.append(f'NOTE: "{msg.interface.nl_summary[:200]}"')

        if annotated:
            analysis = self._semantic_analysis(msg)
            if analysis:
                lines.append("")
                lines.append(analysis)

        return "\n".join(lines)


    def _header(self, msg: Message) -> str:
        rb = msg.role_binding
        priority_str = f" | priority={rb.priority:.2f}" if rb.priority != 0.8 else ""
        ttl_str = f" | ttl={rb.ttl_ms}ms" if rb.ttl_ms > 0 else ""
        turn_str = f" | {rb.turn_idx}" if rb.turn_idx != 1 else ""
        return (
            f"@{rb.sender_id} [{rb.role_type.value}] "
            f"→ {rb.target_id} "
            f"| {rb.session_id}"
            f"{turn_str}{priority_str}{ttl_str}"
        )


    def _act_block(self, msg: Message, annotated: bool) -> str:
        sg = msg.semantic_graph
        st = msg.state_tensor

        acts = sorted(sg.speech_act.acts, key=lambda a: a.value)
        act_str = "+".join(a.value for a in acts)
        if annotated:
            act_comments = [_SPEECH_ACT_COMMENT.get(a, a.value) for a in acts]
            act_str += f"  # {' + '.join(act_comments)}"

        lines = [f"{act_str} {{"]

        conf_comment = ""
        if annotated:
            unc_c = _UNCERT_COMMENT.get(st.uncertainty_type, "")
            delta_c = "  # delta (incrementale)" if st.delta_flag else ""
            conf_comment = unc_c or delta_c
        lines.append(
            f"  CONFIDENCE {st.confidence:.2f}"
            + (f" | uncertainty={st.uncertainty_type.value}" if st.uncertainty_type != UncertaintyType.NONE else "")
            + conf_comment
        )
        lines.append("")

        by_type: dict[NodeType, list[Node]] = {}
        for n in sg.nodes:
            by_type.setdefault(n.type, []).append(n)

        type_order = [
            NodeType.GOAL, NodeType.ACTION, NodeType.CONSTRAINT,
            NodeType.ENTITY, NodeType.CONCEPT, NodeType.STATE,
            NodeType.FAILED_STATE, NodeType.PARTIAL_STATE,
            NodeType.RISK, NodeType.QUALITY_DEGRADED, NodeType.SYNC_BARRIER,
        ]
        ordered_types = type_order + [t for t in by_type if t not in type_order]

        for ntype in ordered_types:
            if ntype not in by_type:
                continue
            nodes = by_type[ntype]
            if annotated:
                lines.append(f"  # ── {ntype.value} ──")
            for node in nodes:
                lines.extend(self._node_block(node, sg, annotated))
                lines.append("")

        lines.append("}")
        return "\n".join(lines)


    def _node_block(self, node: Node, sg: SemanticGraph, annotated: bool) -> list[str]:
        lines = []
        value_display = node.value.replace("_", " ")
        status_comment = _STATUS_COMMENT.get(node.status, "") if annotated else ""

        lines.append(f'  {node.type.value} "{value_display}"{status_comment}')

        if node.salience != 0.9:
            lines.append(f"    salience = {node.salience:.2f}")
        if node.status != NodeStatus.ACTIVE:
            lines.append(f"    status = {node.status.value}")
        if node.authority_required:
            lines.append(f"    authority = {node.authority_required}")
        if node.quality_threshold != 1.0:
            lines.append(f"    quality_min = {node.quality_threshold:.2f}")
        if node.type == NodeType.SYNC_BARRIER and node.sync_sessions:
            sessions_str = " | ".join(node.sync_sessions)
            lines.append(f"    sessions = [{sessions_str}]")
            if node.sync_timeout_ms:
                lines.append(f"    timeout = {node.sync_timeout_ms}")
            if node.sync_action:
                lines.append(f'    sync_action = "{node.sync_action}"')

        out_edges = sg.edges_from(node.node_id)
        for edge in out_edges:
            lines.append(self._edge_line(edge, node, sg, annotated))

        return lines


    def _edge_line(self, edge: Edge, from_node: Node,
                   sg: SemanticGraph, annotated: bool) -> str:
        rel = edge.rel_type.value
        rel_comment = _REL_COMMENT.get(edge.rel_type, rel.lower()) if annotated else ""

        if isinstance(edge.to_id, int):
            target_node = sg.get_node(edge.to_id)
            target = f'"{target_node.value.replace("_", " ")}"' if target_node else str(edge.to_id)
        else:
            target = str(edge.to_id)

        attrs = []
        if edge.weight != 1.0:
            attrs.append(f"weight={edge.weight:.2f}")
        if edge.quality != 1.0:
            attrs.append(f"quality={edge.quality:.2f}")
        if edge.condition_ref:
            attrs.append(f"if {edge.condition_ref}")

        attr_str = "  " + "  ".join(attrs) if attrs else ""
        comment = f"  # {rel_comment}" if annotated and rel_comment else ""

        return f"    → {rel} {target}{attr_str}{comment}"


    def _semantic_analysis(self, msg: Message) -> str:
        sg = msg.semantic_graph
        lines: list[str] = []

        target_ids = {e.to_id for e in sg.edges if isinstance(e.to_id, int)}
        for tid in target_ids:
            res = resolve_concurrent_edges(sg, tid)
            if res is None:
                continue
            node = sg.get_node(tid)
            name = node.value if node else str(tid)
            if res.blocked:
                lines.append(f"#  ANALISI: \"{name}\" è BLOCCATO (regola §5.5.1.1)")
                lines.append(f"#    {res.description}")
            elif res.requires_query:
                lines.append(f"#  ANALISI: \"{name}\" — pesi ambigui, richiede QUERY (§5.5.1.3)")
            elif res.requires_risk_node:
                lines.append(f"#   ANALISI: \"{name}\" — quality degradata ({res.quality_effective:.0%}), serve RISK node")

        for node in sg.nodes:
            if node.type == NodeType.CONSTRAINT:
                chk = check_constraint_quality(sg, node.node_id)
                if not chk.satisfied:
                    lines.append(
                        f"#   ANALISI: constraint \"{node.value}\" VIOLATO "
                        f"(quality {chk.quality_found:.0%} < {chk.quality_required:.0%}) — §5.5.9"
                    )

        vresult = MessageValidator().validate(msg)
        if vresult.errors:
            for e in vresult.errors:
                lines.append(f"#   ERRORE: {e}")
        if not lines:
            lines.append("#   Nessun conflitto semantico rilevato.")

        return "\n".join(lines)


    @classmethod
    def from_json(cls, raw: str) -> tuple[Optional[Message], str]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, f"JSON non valido: {e}"
        try:
            return cls._parse_json_dict(data), ""
        except Exception as e:
            return None, f"Errore parsing: {e}"

    @classmethod
    def _parse_json_dict(cls, d: dict) -> Message:
        from rlp4ai.model import (
            RoleBinding, StateTensor, SemanticGraph, InterfaceLayer,
            Message, Node, Edge, NodeType, NodeStatus, RelType,
            RoleType, SpeechActSet, UncertaintyType, Compression, Modality,
        )

        def safe(enum_cls, val, default):
            try:
                return enum_cls(str(val).strip().upper())
            except ValueError:
                return default

        l1 = d.get("LAYER_1") or d.get("layer1") or {}
        rb = RoleBinding(
            sender_id=l1.get("sender_id", "UNKNOWN"),
            role_type=safe(RoleType, l1.get("role_type", "COORDINATOR"), RoleType.COORDINATOR),
            target_id=l1.get("target_id", "BROADCAST"),
            session_id=l1.get("session_id", "SESSION-001"),
            turn_idx=int(l1.get("turn_idx", 1)),
            ttl_ms=int(str(l1.get("ttl_ms", l1.get("ttl", "0"))).replace("ms", "")),
            priority=float(l1.get("priority", 0.8)),
        )

        l2 = d.get("LAYER_2") or d.get("layer2") or {}
        st = StateTensor(
            confidence=float(l2.get("confidence", 0.85)),
            uncertainty_type=safe(UncertaintyType, l2.get("uncertainty_type", "NONE"), UncertaintyType.NONE),
            delta_flag=bool(l2.get("delta_flag", False)),
            compression=safe(Compression, l2.get("compression", "NONE"), Compression.NONE),
            modality=safe(Modality, l2.get("modality", "ABSTRACT"), Modality.ABSTRACT),
        )

        l3 = d.get("LAYER_3") or d.get("layer3") or {}
        nodes = []
        for n in l3.get("nodes", []):
            nid = int(str(n.get("node_id", 0)).strip("[]"))
            nodes.append(Node(
                node_id=nid,
                type=safe(NodeType, n.get("type", "CONCEPT"), NodeType.CONCEPT),
                value=str(n.get("value", "unknown")),
                salience=float(n.get("salience", 0.8)),
                status=safe(NodeStatus, n.get("status", "ACTIVE"), NodeStatus.ACTIVE),
                authority_required=n.get("authority_required"),
                quality_threshold=float(n.get("quality_threshold", 1.0)),
                sync_sessions=n.get("sync_sessions", []),
                sync_timeout_ms=int(n.get("sync_timeout_ms", 0)),
                sync_action=n.get("sync_action"),
            ))

        edges = []
        for e in l3.get("edges", []):
            fr = e.get("from", e.get("from_id", 0))
            to = e.get("to", e.get("to_id", 0))
            try:
                fr = int(str(fr).strip("[]"))
            except ValueError:
                fr = str(fr)
            try:
                to = int(str(to).strip("[]"))
            except ValueError:
                to = str(to)
            edges.append(Edge(
                from_id=fr, to_id=to,
                rel_type=safe(RelType, e.get("rel_type", "ENABLES"), RelType.ENABLES),
                weight=float(e.get("weight", 1.0)),
                quality=float(e.get("quality", 1.0)),
                condition_ref=e.get("condition_ref"),
            ))

        sa_raw = str(l3.get("speech_act", "INFORM")).replace(" ", "")
        sg = SemanticGraph(
            nodes=nodes, edges=edges,
            speech_act=SpeechActSet.parse(sa_raw),
            intent_description=str(l3.get("intent_vector", "")),
        )

        l4 = d.get("LAYER_4") or d.get("layer4") or {}
        il = InterfaceLayer(
            nl_summary=l4.get("nl_summary", ""),
            target_format=l4.get("target_format", "HUMAN"),
        )
        return Message(rb, st, sg, il)
