from __future__ import annotations

import re
import sys
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlp4ai.model import (
    Compression, Edge, InterfaceLayer, Message, Modality, Node,
    NodeStatus, NodeType, RelType, RoleBinding, RoleType,
    SemanticGraph, SpeechAct, SpeechActSet, StateTensor, UncertaintyType,
)
from rlp4ai.engine import MessageValidator


TK_HEADER    = "HEADER"
TK_ACT       = "ACT"
TK_NODE_TYPE = "NODE_TYPE"
TK_REL_TYPE  = "REL_TYPE"
TK_STATUS    = "STATUS"
TK_UNCERT    = "UNCERT"
TK_STRING    = "STRING"
TK_FLOAT     = "FLOAT"
TK_INT       = "INT"
TK_ARROW     = "ARROW"
TK_PIPE      = "PIPE"
TK_EQ        = "EQ"
TK_LBRACKET  = "LBRACKET"
TK_RBRACKET  = "RBRACKET"
TK_LBRACE    = "LBRACE"
TK_RBRACE    = "RBRACE"
TK_AT        = "AT"
TK_PLUS      = "PLUS"
TK_KW        = "KW"
TK_IDENT     = "IDENT"
TK_COMMENT   = "COMMENT"
TK_NEWLINE   = "NEWLINE"
TK_EOF       = "EOF"

_NODE_TYPES = {
    "GOAL", "ACTION", "CONSTRAINT", "ENTITY", "CONCEPT", "STATE",
    "FAILED_STATE", "PARTIAL_STATE", "RISK", "QUALITY_DEGRADED", "SYNC_BARRIER",
}
_ACT_TYPES = {
    "DELEGATE", "INFORM", "WARN", "COMMIT", "DENY", "QUERY", "PROPOSE",
    "PROPOSE_ALTERNATIVE", "CONFIRM", "ACKNOWLEDGE", "OVERRIDE_REQUEST",
    "OVERRIDE_GRANT", "TIMEOUT_ESCALATE", "EXCEPTION_ACCEPT",
    "BARRIER_EXTENSION_REQUEST", "APPROVES", "REJECTS",
}
_REL_TYPES = {
    "ENABLES", "BLOCKS", "CAUSES", "REQUIRES", "SUPPORTS", "RESOLVES",
    "MITIGATES", "MODIFIES", "DEGRADES", "CONTRADICTS", "IS_A",
    "HAS_PROPERTY", "TEMPORAL_BEFORE", "TEMPORAL_AFTER", "PART_OF",
    "EQUIVALENT", "CONDITIONAL_ENABLES", "CONDITIONAL_BLOCKS",
}
_STATUS_KW  = {"ACTIVE", "BLOCKED", "SUPERSEDED", "PENDING", "RESOLVED", "BYPASSED"}
_UNCERT_KW  = {"EPISTEMIC", "ALEATORIC", "MIXED", "NONE"}
_ATTR_KW    = {"salience", "status", "authority", "quality_min",
               "sessions", "timeout", "sync_action", "weight", "quality",
               "uncertainty", "priority", "ttl", "delta"}


@dataclass
class Token:
    type: str
    value: str
    line: int


class LexError(Exception):
    pass

class ParseError(Exception):
    pass


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    i = 0
    src = source

    while i < len(src):
        if src[i] == "\n":
            line += 1
            i += 1
            continue

        if src[i] in " \t\r":
            i += 1
            continue

        if src[i] == "#":
            while i < len(src) and src[i] != "\n":
                i += 1
            continue

        if src[i] == '"':
            j = i + 1
            while j < len(src) and src[j] != '"':
                if src[j] == "\n":
                    line += 1
                j += 1
            if j >= len(src):
                raise LexError(f"Riga {line}: stringa non chiusa")
            tokens.append(Token(TK_STRING, src[i+1:j], line))
            i = j + 1
            continue

        if src[i:i+2] == "→" or src[i:i+2] == "->":
            tokens.append(Token(TK_ARROW, "→", line))
            i += 2 if src[i] == "-" else len("→")
            continue
        if src[i] == "→":
            tokens.append(Token(TK_ARROW, "→", line))
            i += len("→")
            continue
        if src[i] == "|":
            tokens.append(Token(TK_PIPE, "|", line))
            i += 1
            continue
        if src[i] == "=":
            tokens.append(Token(TK_EQ, "=", line))
            i += 1
            continue
        if src[i] == "[":
            tokens.append(Token(TK_LBRACKET, "[", line))
            i += 1
            continue
        if src[i] == "]":
            tokens.append(Token(TK_RBRACKET, "]", line))
            i += 1
            continue
        if src[i] == "{":
            tokens.append(Token(TK_LBRACE, "{", line))
            i += 1
            continue
        if src[i] == "}":
            tokens.append(Token(TK_RBRACE, "}", line))
            i += 1
            continue
        if src[i] == "@":
            tokens.append(Token(TK_AT, "@", line))
            i += 1
            continue
        if src[i] == "+":
            tokens.append(Token(TK_PLUS, "+", line))
            i += 1
            continue
        if src[i] == ":":
            tokens.append(Token("COLON", ":", line))
            i += 1
            continue

        m = re.match(r"(\d+\.\d+|\d+)", src[i:])
        if m:
            val = m.group(1)
            tk_type = TK_FLOAT if "." in val else TK_INT
            tokens.append(Token(tk_type, val, line))
            i += len(val)
            continue

        m = re.match(r"([A-Za-z_][A-Za-z0-9_\-]*)", src[i:])
        if m:
            word = m.group(1)
            upper = word.upper()
            if upper in _NODE_TYPES:
                tokens.append(Token(TK_NODE_TYPE, upper, line))
            elif upper in _ACT_TYPES:
                tokens.append(Token(TK_ACT, upper, line))
            elif upper in _REL_TYPES:
                tokens.append(Token(TK_REL_TYPE, upper, line))
            elif upper in _STATUS_KW:
                tokens.append(Token(TK_STATUS, upper, line))
            elif upper in _UNCERT_KW:
                tokens.append(Token(TK_UNCERT, upper, line))
            elif word.lower() in _ATTR_KW:
                tokens.append(Token(TK_KW, word.lower(), line))
            else:
                tokens.append(Token(TK_IDENT, word, line))
            i += len(word)
            continue

        i += 1

    tokens.append(Token(TK_EOF, "", line))
    return tokens



class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self._node_counter = 0
        self._nodes: list[Node] = []
        self._edges: list[Edge] = []
        self._last_node_id: Optional[int] = None

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, expected_type: Optional[str] = None) -> Token:
        tok = self.tokens[self.pos]
        if expected_type and tok.type != expected_type:
            raise ParseError(
                f"Riga {tok.line}: atteso {expected_type}, trovato "
                f"{tok.type}({tok.value!r})"
            )
        self.pos += 1
        return tok

    def match(self, *types) -> bool:
        return self.peek().type in types

    def _next_node_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid


    def parse(self) -> dict:
        result = {
            "sender_id": "USER",
            "role_type": RoleType.COORDINATOR,
            "target_id": "BROADCAST",
            "session_id": f"SESSION-{uuid.uuid4().hex[:6].upper()}",
            "turn_idx": 1,
            "priority": 0.8,
            "ttl_ms": 0,
            "speech_acts": [],
            "confidence": 0.85,
            "uncertainty": UncertaintyType.NONE,
            "delta_flag": False,
            "note": "",
        }

        if self.match(TK_AT):
            self._parse_header(result)

        while self.match(TK_ACT):
            acts, nodes, edges = self._parse_act_block(result)
            result["speech_acts"].extend(acts)
            self._nodes.extend(nodes)
            self._edges.extend(edges)

        # NOTE: "..."
        if self.match(TK_IDENT) and self.peek().value.upper() == "NOTE":
            self.consume()
            self.consume("COLON")
            tok = self.consume(TK_STRING)
            result["note"] = tok.value

        result["nodes"] = self._nodes
        result["edges"] = self._edges

        if not result["speech_acts"]:
            result["speech_acts"] = [SpeechAct.INFORM]

        return result


    def _parse_header(self, result: dict) -> None:
        self.consume(TK_AT)
        sender = self.consume(TK_IDENT).value
        result["sender_id"] = sender

        if self.match(TK_LBRACKET):
            self.consume(TK_LBRACKET)
            role_tok = self.consume(TK_IDENT)
            result["role_type"] = _parse_role(role_tok.value)
            self.consume(TK_RBRACKET)

        if self.match(TK_ARROW):
            self.consume(TK_ARROW)
            target = self._parse_ident_or_broadcast()
            result["target_id"] = target

        while self.match(TK_PIPE):
            self.consume(TK_PIPE)
            if self.match(TK_KW):
                kw = self.consume(TK_KW).value
                self.consume(TK_EQ)
                if kw == "priority":
                    result["priority"] = float(self.consume(TK_FLOAT).value)
                elif kw == "ttl":
                    result["ttl_ms"] = int(self.consume(TK_INT).value)
                    if self.match(TK_IDENT) and self.peek().value.lower() == "ms":
                        self.consume(TK_IDENT)
                elif kw == "delta":
                    result["delta_flag"] = True
            elif self.match(TK_IDENT):
                result["session_id"] = self.consume(TK_IDENT).value
            elif self.match(TK_INT):
                result["turn_idx"] = int(self.consume(TK_INT).value)

    def _parse_ident_or_broadcast(self) -> str:
        parts = []
        if self.match(TK_IDENT):
            parts.append(self.consume(TK_IDENT).value)
        return "-".join(parts) if parts else "BROADCAST"


    def _parse_act_block(self, result: dict) -> tuple:
        acts: list[SpeechAct] = []
        nodes: list[Node] = []
        edges: list[Edge] = []
        local_node_counter = [self._node_counter]

        act_tok = self.consume(TK_ACT)
        acts.append(SpeechAct(act_tok.value))
        while self.match(TK_PLUS):
            self.consume(TK_PLUS)
            acts.append(SpeechAct(self.consume(TK_ACT).value))

        self.consume(TK_LBRACE)

        while not self.match(TK_RBRACE) and not self.match(TK_EOF):
            if self.match(TK_NODE_TYPE):
                node, node_edges = self._parse_node_stmt()
                nodes.append(node)
                edges.extend(node_edges)
            elif self.match(TK_KW) and self.peek().value == "confidence":
                conf, uncert = self._parse_confidence_stmt()
                result["confidence"] = conf
                result["uncertainty"] = uncert
            elif self.match(TK_IDENT) and self.peek().value.upper() == "CONFIDENCE":
                self.consume()
                conf_tok = self.consume(TK_FLOAT)
                result["confidence"] = float(conf_tok.value)
                if self.match(TK_PIPE):
                    self.consume(TK_PIPE)
                    if self.match(TK_KW):
                        self.consume(TK_KW) 
                        self.consume(TK_EQ)
                        result["uncertainty"] = UncertaintyType(
                            self.consume(TK_UNCERT).value
                        )
            else:
                self.pos += 1

        self.consume(TK_RBRACE)
        return acts, nodes, edges


    def _parse_node_stmt(self) -> tuple[Node, list[Edge]]:
        ntype_tok = self.consume(TK_NODE_TYPE)
        ntype = NodeType(ntype_tok.value)
        value_tok = self.consume(TK_STRING)
        value = value_tok.value.replace(" ", "_")

        node_id = self._next_node_id()
        self._last_node_id = node_id

        salience = 0.9
        status = NodeStatus.ACTIVE
        authority = None
        quality_min = 1.0
        sync_sessions: list[str] = []
        sync_timeout = 0
        sync_action = None

        edges: list[Edge] = []

        while self.match(TK_KW) or self.match(TK_ARROW):
            if self.match(TK_ARROW):
                edge = self._parse_edge_stmt(node_id)
                if edge:
                    edges.append(edge)
                continue

            kw = self.consume(TK_KW).value
            self.consume(TK_EQ)

            if kw == "salience":
                salience = float(self.consume(TK_FLOAT).value)
            elif kw == "status":
                status = NodeStatus(self.consume(TK_STATUS).value)
            elif kw == "authority":
                authority = self.consume(TK_IDENT).value
            elif kw == "quality_min":
                quality_min = float(self.consume(TK_FLOAT).value)
            elif kw == "sessions":
                self.consume(TK_LBRACKET)
                sync_sessions = []
                while not self.match(TK_RBRACKET):
                    if self.match(TK_IDENT):
                        sync_sessions.append(self.consume(TK_IDENT).value)
                    elif self.match(TK_PIPE):
                        self.consume(TK_PIPE)
                self.consume(TK_RBRACKET)
            elif kw == "timeout":
                sync_timeout = int(self.consume(TK_INT).value)
            elif kw == "sync_action":
                sync_action = self.consume(TK_STRING).value

        node = Node(
            node_id=node_id,
            type=ntype,
            value=value,
            salience=salience,
            status=status,
            authority_required=authority,
            quality_threshold=quality_min,
            sync_sessions=sync_sessions,
            sync_timeout_ms=sync_timeout,
            sync_action=sync_action,
        )
        return node, edges


    def _parse_edge_stmt(self, from_id: int) -> Optional[Edge]:
        self.consume(TK_ARROW)

        if not self.match(TK_REL_TYPE):
            rel_type = RelType.ENABLES
        else:
            rel_tok = self.consume(TK_REL_TYPE)
            rel_type = RelType(rel_tok.value)

        to_id: object
        condition_ref: Optional[str] = None

        if self.match(TK_STRING):
            target_val = self.consume(TK_STRING).value.replace(" ", "_").lower()
            to_id = f"__RESOLVE__{target_val}"
        elif self.match(TK_AT):
            self.consume(TK_AT)
            session = self.consume(TK_IDENT).value
            self.consume("COLON")
            turn_tok = self.consume(TK_IDENT)
            turn_raw = turn_tok.value 
            if re.match(r"turn\d+$", turn_raw):
                turn = re.search(r"\d+$", turn_raw).group()
            else:
                turn = self.consume(TK_INT).value
            self.consume("COLON")
            node_tok = self.consume(TK_IDENT)
            node_raw = node_tok.value  
            if re.match(r"node\d+$", node_raw):
                node = re.search(r"\d+$", node_raw).group()
            else:
                node = self.consume(TK_INT).value
            to_id = f"@{session}:turn{turn}:node{node}"
        elif self.match(TK_INT):
            to_id = int(self.consume(TK_INT).value)
        else:
            return None

        weight = 1.0
        quality = 1.0

        while self.match(TK_KW):
            kw = self.consume(TK_KW).value
            self.consume(TK_EQ)
            if kw == "weight":
                weight = float(self.consume(TK_FLOAT).value)
            elif kw == "quality":
                quality = float(self.consume(TK_FLOAT).value)

        if self.match(TK_IDENT) and self.peek().value.lower() == "if":
            self.consume()
            if self.match(TK_AT):
                self.consume(TK_AT)
                session = self.consume(TK_IDENT).value
                self.consume("COLON")
                self.consume(TK_IDENT)
                turn = self.consume(TK_INT).value
                self.consume("COLON")
                self.consume(TK_IDENT)
                node = self.consume(TK_INT).value
                condition_ref = f"@{session}:turn{turn}:node{node}"
            elif self.match(TK_INT):
                condition_ref = str(self.consume(TK_INT).value)

        return Edge(
            from_id=from_id,
            to_id=to_id,
            rel_type=rel_type,
            weight=weight,
            quality=quality,
            condition_ref=condition_ref,
        )

    def _parse_confidence_stmt(self) -> tuple[float, UncertaintyType]:
        self.consume(TK_KW) 
        self.consume(TK_EQ)
        conf = float(self.consume(TK_FLOAT).value)
        uncert = UncertaintyType.NONE
        if self.match(TK_PIPE):
            self.consume(TK_PIPE)
            if self.match(TK_KW):
                self.consume(TK_KW) 
                self.consume(TK_EQ)
                uncert = UncertaintyType(self.consume(TK_UNCERT).value)
        return conf, uncert



class CompileError(Exception):
    pass


def compile_plai(source: str) -> tuple[Message, list[str]]:

    try:
        tokens = tokenize(source)
    except LexError as e:
        raise CompileError(f"Errore lessicale: {e}")

    try:
        parser = Parser(tokens)
        result = parser.parse()
    except ParseError as e:
        raise CompileError(f"Errore di sintassi: {e}")

    nodes: list[Node] = result["nodes"]
    edges: list[Edge] = result["edges"]

    value_to_id = {n.value: n.node_id for n in nodes}
    resolved_edges = []
    for e in edges:
        if isinstance(e.to_id, str) and e.to_id.startswith("__RESOLVE__"):
            target_val = e.to_id[len("__RESOLVE__"):]
            if target_val in value_to_id:
                e = Edge(
                    from_id=e.from_id,
                    to_id=value_to_id[target_val],
                    rel_type=e.rel_type,
                    weight=e.weight,
                    quality=e.quality,
                    condition_ref=e.condition_ref,
                )
            else:
                e = Edge(
                    from_id=e.from_id,
                    to_id=target_val,
                    rel_type=e.rel_type,
                    weight=e.weight,
                    quality=e.quality,
                    condition_ref=e.condition_ref,
                )
        resolved_edges.append(e)

    rb = RoleBinding(
        sender_id=result["sender_id"],
        role_type=result["role_type"],
        target_id=result["target_id"],
        session_id=result["session_id"],
        turn_idx=result["turn_idx"],
        ttl_ms=result["ttl_ms"],
        priority=result["priority"],
    )

    st = StateTensor(
        confidence=result["confidence"],
        uncertainty_type=result["uncertainty"],
        delta_flag=result["delta_flag"],
        compression=Compression.NONE,
        modality=Modality.LINGUISTIC,
    )

    speech_act_set = SpeechActSet(frozenset(result["speech_acts"]))

    sg = SemanticGraph(
        nodes=nodes,
        edges=resolved_edges,
        speech_act=speech_act_set,
        intent_description=result.get("note", ""),
    )

    il = InterfaceLayer(
        nl_summary=result.get("note", ""),
        target_format="HUMAN",
        verbosity="STANDARD",
    )

    msg = Message(rb, st, sg, il)

    validator = MessageValidator()
    vresult = validator.validate(msg)
    warnings = vresult.warnings[:]
    if not vresult.valid:
        raise CompileError(
            "Messaggio non valido dopo compilazione:\n" +
            "\n".join(f"  • {e}" for e in vresult.errors)
        )

    return msg, warnings



def _parse_role(s: str) -> RoleType:
    mapping = {
        "COORDINATOR": RoleType.COORDINATOR,
        "EXECUTOR": RoleType.EXECUTOR,
        "VERIFIER": RoleType.VERIFIER,
        "OBSERVER": RoleType.OBSERVER,
        "HYBRID": RoleType.HYBRID,
        "INCIDENT_COMMANDER": RoleType.INCIDENT_COMMANDER,
        "MIGRATION_COMMANDER": RoleType.MIGRATION_COMMANDER,
        "CLUSTER_MANAGER": RoleType.CLUSTER_MANAGER,
    }
    return mapping.get(s.upper(), RoleType.COORDINATOR)
