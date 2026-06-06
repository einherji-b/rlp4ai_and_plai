from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .model import (
    CrossRef, Message, Node, NodeStatus, NodeType,
    RelType, SpeechAct,
)


class SessionStore:

    def __init__(self) -> None:
        self._messages: dict[str, dict[int, Message]] = defaultdict(dict)
        self._node_state: dict[str, dict[int, Node]] = defaultdict(dict)
        self._turn_order: dict[str, list[int]] = defaultdict(list)
        self._barriers: dict[str, "SyncBarrierState"] = {}


    def ingest(self, msg: Message) -> None:

        sid = msg.session_id
        tidx = msg.turn_idx

        if tidx in self._messages[sid]:
            raise ValueError(
                f"turn_idx={tidx} già presente per session={sid}."
            )
        self._messages[sid][tidx] = msg

        turns = self._turn_order[sid]
        turns.append(tidx)
        turns.sort()

        self._apply_delta(msg)

    def _apply_delta(self, msg: Message) -> None:

        sid = msg.session_id
        g = msg.semantic_graph

        for node in g.nodes:

            self._node_state[sid][node.node_id] = node

        for edge in g.edges:
            if edge.rel_type == RelType.MODIFIES:
                target = edge.to_id
                if isinstance(target, str) and CrossRef.is_cross_ref(target):
                    cr = CrossRef.parse(target)
                    if cr.session_id == sid:
                        old_node = self._node_state[sid].get(cr.node_id)
                        if old_node is not None:
                            old_node.status = NodeStatus.SUPERSEDED

        for edge in g.edges:
            if edge.rel_type == RelType.RESOLVES:
                target = edge.to_id
                if isinstance(target, str) and CrossRef.is_cross_ref(target):
                    cr = CrossRef.parse(target)
                    old_node = self._get_node_by_crossref(cr)
                    if old_node is not None and old_node.status == NodeStatus.ACTIVE:
                        old_node.status = NodeStatus.RESOLVED

        self._update_sync_barriers(msg)

    def _update_sync_barriers(self, msg: Message) -> None:
        sa = msg.semantic_graph.speech_act
        if not sa.commits():
            return
        for barrier in self._barriers.values():
            if barrier.status == "ACTIVE":
                barrier.register_commit(msg.session_id)


    def get_node(self, ref: CrossRef) -> Optional[Node]:
 
        node = self._get_node_by_crossref(ref)
        if node is None:
            return None
        if node.status == NodeStatus.SUPERSEDED:
            return None  
        return node

    def _get_node_by_crossref(self, ref: CrossRef) -> Optional[Node]:
        session_nodes = self._node_state.get(ref.session_id, {})
        return session_nodes.get(ref.node_id)

    def get_message(self, session_id: str, turn_idx: int) -> Optional[Message]:
        return self._messages.get(session_id, {}).get(turn_idx)

    def current_nodes(self, session_id: str) -> dict[int, Node]:
        return dict(self._node_state.get(session_id, {}))

    def history(self, session_id: str) -> list[Message]:
        turns = self._turn_order.get(session_id, [])
        msgs = self._messages.get(session_id, {})
        return [msgs[t] for t in turns]


    def register_barrier(self, barrier_key: str, node: Node) -> "SyncBarrierState":
        state = SyncBarrierState(
            key=barrier_key,
            sync_sessions=list(node.sync_sessions),
            sync_timeout_ms=node.sync_timeout_ms,
            sync_action=node.sync_action or "",
        )
        self._barriers[barrier_key] = state
        return state

    def get_barrier(self, barrier_key: str) -> Optional["SyncBarrierState"]:
        return self._barriers.get(barrier_key)


    def validate_cross_ref(self, ref_str: str, current_turn: int) -> tuple[bool, str]:

        try:
            cr = CrossRef.parse(ref_str)
        except ValueError as e:
            return False, str(e)

        if cr.turn_idx > current_turn:
            return False, (
                f"Riferimento a turno futuro: turn_idx={cr.turn_idx} > "
                f"current_turn={current_turn}."
            )

        node = self._get_node_by_crossref(cr)
        if node is None:
            return False, f"Nodo {ref_str} non trovato nella storia della sessione."

        if node.status == NodeStatus.SUPERSEDED:
            return False, (
                f"Nodo {ref_str} è SUPERSEDED — obbligatorio QUERY (§5.5.3)."
            )

        return True, f"Riferimento valido → nodo status={node.status.value}."



@dataclass
class SyncBarrierState:
    key: str
    sync_sessions: list[str]
    sync_timeout_ms: int
    sync_action: str
    status: str = "ACTIVE"  
    committed: set[str] = field(default_factory=set)
    extension_granted_ms: int = 0

    def register_commit(self, session_id: str) -> bool:

        if self.status != "ACTIVE":
            return False
        self.committed.add(session_id)
        required = set(self.sync_sessions)
        if required.issubset(self.committed):
            self.status = "RESOLVED"
            return True
        return False

    def remaining(self) -> set[str]:
        return set(self.sync_sessions) - self.committed

    def apply_extension(self, extension_ms: int) -> None:
        self.extension_timeout_ms = self.sync_timeout_ms + extension_ms
        self.extension_granted_ms = extension_ms

    def block(self) -> None:
        self.status = "BLOCKED"

    def effective_timeout_ms(self) -> int:
        return self.sync_timeout_ms + self.extension_granted_ms

    def __repr__(self) -> str:
        remaining = self.remaining()
        return (
            f"SyncBarrier({self.key!r} status={self.status} "
            f"committed={self.committed} remaining={remaining})"
        )
