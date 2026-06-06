from __future__ import annotations

import re
import time
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union


class RoleType(str, Enum):
    COORDINATOR          = "COORDINATOR"
    EXECUTOR             = "EXECUTOR"
    VERIFIER             = "VERIFIER"
    OBSERVER             = "OBSERVER"
    HYBRID               = "HYBRID"
    INCIDENT_COMMANDER   = "INCIDENT_COMMANDER"
    MIGRATION_COMMANDER  = "MIGRATION_COMMANDER"
    CLUSTER_MANAGER      = "CLUSTER_MANAGER"
    RESPONDER_PRIMARY    = "RESPONDER_PRIMARY"
    RESPONDER_SECONDARY  = "RESPONDER_SECONDARY"
    INTEGRITY_VERIFIER   = "INTEGRITY_VERIFIER"
    ANALYST              = "ANALYST"


class NodeType(str, Enum):
    ENTITY          = "ENTITY"
    CONCEPT         = "CONCEPT"
    ACTION          = "ACTION"
    CONSTRAINT      = "CONSTRAINT"
    GOAL            = "GOAL"
    STATE           = "STATE"
    FAILED_STATE    = "FAILED_STATE"
    PARTIAL_STATE   = "PARTIAL_STATE"
    RISK            = "RISK"
    QUALITY_DEGRADED = "QUALITY_DEGRADED"
    SYNC_BARRIER    = "SYNC_BARRIER"


class NodeStatus(str, Enum):
    ACTIVE      = "ACTIVE"
    BLOCKED     = "BLOCKED"
    SUPERSEDED  = "SUPERSEDED"
    PENDING     = "PENDING"
    RESOLVED    = "RESOLVED"
    BYPASSED    = "BYPASSED"   


class RelType(str, Enum):
    CAUSES              = "CAUSES"
    REQUIRES            = "REQUIRES"
    ENABLES             = "ENABLES"
    BLOCKS              = "BLOCKS"
    IS_A                = "IS_A"
    HAS_PROPERTY        = "HAS_PROPERTY"
    TEMPORAL_BEFORE     = "TEMPORAL_BEFORE"
    TEMPORAL_AFTER      = "TEMPORAL_AFTER"
    PART_OF             = "PART_OF"
    EQUIVALENT          = "EQUIVALENT"
    CONTRADICTS         = "CONTRADICTS"
    SUPPORTS            = "SUPPORTS"
    RESOLVES            = "RESOLVES"
    MITIGATES           = "MITIGATES"
    MODIFIES            = "MODIFIES"
    DEGRADES            = "DEGRADES"
    CONDITIONAL_ENABLES = "CONDITIONAL_ENABLES"
    CONDITIONAL_BLOCKS  = "CONDITIONAL_BLOCKS"

    @property
    def is_blocking(self) -> bool:
        return self in (RelType.BLOCKS, RelType.CONDITIONAL_BLOCKS)

    @property
    def is_enabling(self) -> bool:
        return self in (RelType.ENABLES, RelType.CONDITIONAL_ENABLES,
                        RelType.SUPPORTS, RelType.RESOLVES)


class SpeechAct(str, Enum):
    INFORM                    = "INFORM"
    REQUEST                   = "REQUEST"
    CONFIRM                   = "CONFIRM"
    DENY                      = "DENY"
    PROPOSE                   = "PROPOSE"
    DELEGATE                  = "DELEGATE"
    QUERY                     = "QUERY"
    ACKNOWLEDGE               = "ACKNOWLEDGE"
    WARN                      = "WARN"
    COMMIT                    = "COMMIT"
    PROPOSE_ALTERNATIVE       = "PROPOSE_ALTERNATIVE"
    APPROVES                  = "APPROVES"
    REJECTS                   = "REJECTS"
    OVERRIDE_REQUEST          = "OVERRIDE_REQUEST"
    OVERRIDE_GRANT            = "OVERRIDE_GRANT"
    TIMEOUT_ESCALATE          = "TIMEOUT_ESCALATE"
    EXCEPTION_ACCEPT          = "EXCEPTION_ACCEPT"
    BARRIER_EXTENSION_REQUEST = "BARRIER_EXTENSION_REQUEST"

    @property
    def implies_block(self) -> bool:
        return self == SpeechAct.DENY

    @property
    def implies_warning(self) -> bool:
        return self == SpeechAct.WARN


class UncertaintyType(str, Enum):
    EPISTEMIC = "EPISTEMIC"   
    ALEATORIC = "ALEATORIC"  
    MIXED     = "MIXED"
    NONE      = "NONE"


class Compression(str, Enum):
    NONE = "NONE"
    PCA  = "PCA"
    VQ   = "VQ"
    HASH = "HASH"


class Modality(str, Enum):
    LINGUISTIC = "LINGUISTIC"
    VISUAL     = "VISUAL"
    MIXED      = "MIXED"
    ABSTRACT   = "ABSTRACT"


@dataclass(frozen=True)
class SpeechActSet:
    acts: frozenset[SpeechAct]

    @classmethod
    def parse(cls, raw: str) -> "SpeechActSet":
        parts = [s.strip() for s in raw.split("+")]
        return cls(frozenset(SpeechAct(p) for p in parts))

    def __contains__(self, act: SpeechAct) -> bool:
        return act in self.acts

    def blocks(self) -> bool:
        return SpeechAct.DENY in self.acts

    def warns(self) -> bool:
        return SpeechAct.WARN in self.acts

    def commits(self) -> bool:
        return SpeechAct.COMMIT in self.acts

    def is_query(self) -> bool:
        return SpeechAct.QUERY in self.acts

    def __str__(self) -> str:
        return "+".join(sorted(a.value for a in self.acts))

    def __repr__(self) -> str:
        return f"SpeechActSet({str(self)})"


_CROSS_REF_RE = re.compile(
    r"^@(?P<session>[A-Za-z0-9_\-]+):turn(?P<turn>\d+):node(?P<node>\d+)$"
)

@dataclass(frozen=True)
class CrossRef:
    session_id: str
    turn_idx: int
    node_id: int

    @classmethod
    def parse(cls, s: str) -> "CrossRef":
        m = _CROSS_REF_RE.match(s)
        if not m:
            raise ValueError(f"Riferimento cross-turno non valido: '{s}'")
        return cls(
            session_id=m.group("session"),
            turn_idx=int(m.group("turn")),
            node_id=int(m.group("node")),
        )

    def __str__(self) -> str:
        return f"@{self.session_id}:turn{self.turn_idx}:node{self.node_id}"

    @staticmethod
    def is_cross_ref(s: str) -> bool:
        return isinstance(s, str) and s.startswith("@")


@dataclass
class Node:
    node_id: int
    type: NodeType
    value: str
    salience: float = 1.0
    status: NodeStatus = NodeStatus.ACTIVE
    authority_required: Optional[str] = None
    quality_threshold: float = 1.0     
    sync_sessions: list[str] = field(default_factory=list)
    sync_timeout_ms: int = 0
    sync_action: Optional[str] = None
    _committed_sessions: set[str] = field(default_factory=set, repr=False)

    def is_positive(self) -> bool:
        return self.status not in (
            NodeStatus.BLOCKED, NodeStatus.SUPERSEDED, NodeStatus.BYPASSED
        ) and self.type not in (NodeType.FAILED_STATE,)

    def sync_register_commit(self, session_id: str) -> bool:
        self._committed_sessions.add(session_id)
        required = set(self.sync_sessions)
        if required and required.issubset(self._committed_sessions):
            self.status = NodeStatus.RESOLVED
            return True
        return False

    def sync_remaining(self) -> set[str]:
        return set(self.sync_sessions) - self._committed_sessions


@dataclass
class Edge:
    from_id: Union[int, str]     
    to_id: Union[int, str]       
    rel_type: RelType
    weight: float = 1.0
    quality: float = 1.0       
    condition_ref: Optional[str] = None  
    temporal: bool = False

    def condition_ref_is_cross(self) -> bool:
        return self.condition_ref is not None and CrossRef.is_cross_ref(self.condition_ref)

    def condition_ref_is_local(self) -> bool:
        return (self.condition_ref is not None
                and not CrossRef.is_cross_ref(self.condition_ref))


@dataclass
class RoleBinding:
    sender_id: str
    role_type: RoleType
    target_id: str          
    session_id: str
    turn_idx: int
    timestamp_us: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    ttl_ms: int = 0    
    priority: float = 0.5

    def is_expired(self, now_us: Optional[int] = None) -> bool:
        if self.ttl_ms == 0:
            return False
        now = now_us or int(time.time() * 1_000_000)
        elapsed_ms = (now - self.timestamp_us) / 1000
        return elapsed_ms > self.ttl_ms

    def broadcast_scope(self) -> Optional[str]:
        t = self.target_id.upper()
        if not t.startswith("BROADCAST"):
            return None
        if "@SYSTEM" in t:
            return "SYSTEM"
        if "@" in t:
            return self.target_id.split("@", 1)[1]
        return self.session_id  

@dataclass
class StateTensor:
    state_vector: list[float] = field(default_factory=list)
    confidence: float = 1.0
    uncertainty_type: UncertaintyType = UncertaintyType.NONE
    delta_flag: bool = False
    compression: Compression = Compression.NONE
    modality: Modality = Modality.ABSTRACT

    def fingerprint(self) -> str:
        raw = json.dumps(self.state_vector, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()



@dataclass
class SemanticGraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    speech_act: SpeechActSet = field(
        default_factory=lambda: SpeechActSet.parse("INFORM")
    )
    intent_vector: list[float] = field(default_factory=list)
    intent_description: str = ""

    def get_node(self, node_id: int) -> Optional[Node]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def edges_to(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges if e.to_id == node_id]

    def edges_from(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges if e.from_id == node_id]

    def get_sync_barrier(self) -> Optional[Node]:
        for n in self.nodes:
            if n.type == NodeType.SYNC_BARRIER:
                return n
        return None


@dataclass
class InterfaceLayer:
    nl_summary: str = ""
    target_format: str = "HUMAN"
    verbosity: str = "STANDARD"


@dataclass
class Message:
    role_binding: RoleBinding
    state_tensor: StateTensor
    semantic_graph: SemanticGraph
    interface: Optional[InterfaceLayer] = None

    @property
    def session_id(self) -> str:
        return self.role_binding.session_id

    @property
    def turn_idx(self) -> int:
        return self.role_binding.turn_idx

    @property
    def sender_id(self) -> str:
        return self.role_binding.sender_id

    def checksum(self) -> str:
        parts = {
            "sender": self.role_binding.sender_id,
            "session": self.role_binding.session_id,
            "turn": self.role_binding.turn_idx,
            "nodes": [(n.node_id, n.type.value, n.value)
                      for n in self.semantic_graph.nodes],
            "edges": [(e.from_id, e.to_id, e.rel_type.value)
                      for e in self.semantic_graph.edges],
        }
        raw = json.dumps(parts, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()
