================================================================================
RLP4AI AND PLAI
Version 1.0
================================================================================

INDEX
-----
1. Project objectives
2. RLP4AI — what it is and how it works
3. PLAI — what it is and how it works
4. Repository structure
5. How to use RLP4AI (Python examples)
6. How to use PLAI (interactive CLI and file mode)
7. How to run the tests
8. Dependencies


================================================================================
1. PROJECT OBJECTIVES
================================================================================

Natural language is optimized for human-to-human communication: it tolerates
ambiguity, implicit references, and redundancy. When used as a communication
channel between artificial agents, it introduces a structural loss of semantic
information that accumulates with each exchange — a phenomenon documented in
recent literature as "cascading semantic loss"
(Zhou et al., arXiv:2506.02739, 2025).

This repository presents two complementary tools that address the problem from
different angles:

  RLP4AI (Representation Language Protocol for Artificial Intelligence)
  — a structured communication protocol native to artificial agents, which
  replaces natural language with a typed semantic graph format, preserving
  the semantic distinctions relevant to the task.

  PLAI (Prompting Language AI)
  — a human-readable declarative language that acts as an interface to RLP4AI:
  it allows a user to compose RLP4AI messages without writing raw JSON, and to
  read messages produced by agents in annotated form.

The two tools cover three communication channels:

  H2A (Human to Agent): the user writes in PLAI, the compiler produces RLP4AI
  A2H (Agent to Human): the agent produces RLP4AI, the decompiler produces PLAI
  A2A (Agent to Agent): agents exchange RLP4AI messages directly, without PLAI


================================================================================
2. RLP4AI — WHAT IT IS AND HOW IT WORKS
================================================================================

RLP4AI is a four-layer messaging protocol.

LAYER 1 — ROLE-BINDING LAYER
  Identifies sender and receiver and maintains an explicit role in every message.
  Contains: agent ID, role type, session ID, turn index, timestamp, TTL
  (expiry in ms), target ID, priority.
  This layer resolves the "role confusion" problem documented in natural-language-
  based multi-agent systems: the role is never implicit — it is a structural
  part of the message.

LAYER 2 — STATE-TENSOR LAYER
  Transmits the internal semantic state in compressed form. Contains: float state
  vector (reduced projection of the latent space), confidence, uncertainty type
  (EPISTEMIC / ALEATORIC / MIXED / NONE), delta flag (incremental transmission),
  compression mode. The distinction between epistemic uncertainty (lack of data,
  resolvable) and aleatoric uncertainty (intrinsic variability, not resolvable)
  changes the expected response from the receiving agent.

LAYER 3 — SEMANTIC GRAPH LAYER
  The core of the protocol. Encodes relations, dependencies, intentions, and
  constraints as a typed directed graph. Nodes represent concepts, actions,
  goals, constraints, states, risks, and synchronization barriers. Edges
  represent typed semantic relations between nodes (18 types available) with
  explicit weight and quality values. Every message carries a typed communicative
  act (speech_act) that structurally distinguishes between commitment, warning,
  blocking, delegation, and 20 other act types.

  Available node types: GOAL, ACTION, CONSTRAINT, ENTITY, CONCEPT, STATE,
  FAILED_STATE, PARTIAL_STATE, RISK, QUALITY_DEGRADED, SYNC_BARRIER.

  Available relation types: ENABLES, BLOCKS, CAUSES, REQUIRES, SUPPORTS,
  RESOLVES, MITIGATES, MODIFIES, DEGRADES, CONTRADICTS, CONDITIONAL_ENABLES,
  CONDITIONAL_BLOCKS, IS_A, HAS_PROPERTY, TEMPORAL_BEFORE, TEMPORAL_AFTER,
  PART_OF, EQUIVALENT.

LAYER 4 — INTERFACE LAYER
  Optional translation to natural language or other formats. Mandatory for
  agent-to-human communication, omitted in agent-to-agent communication.
  Contains an nl_summary field (free text, max 512 characters) and format and
  verbosity parameters.

OPERATIONAL SEMANTIC RULES

  The Python library automatically applies all rules from the v1.0 specification:

  Rule §5.5.1 — Concurrent weights: when BLOCKS and ENABLES edges point to the
  same node, the system compares weights. If the BLOCKS weight is greater, the
  node is blocked and the system requires WARN or DENY. If the weights differ by
  less than 0.15, the situation is ambiguous and requires QUERY. If ENABLES
  prevails, the node is active but with degraded quality
  (quality_effective = weight_ENABLES x (1 - weight_BLOCKS)).

  Rule §5.5.2 — Conditional edges: CONDITIONAL_ENABLES and CONDITIONAL_BLOCKS
  require a condition_ref field referencing an existing node (local to the
  message or cross-turn). The edge is active only if the referenced node is in
  a positive state.

  Rule §5.5.3 — Cross-turn references: a reference @SESSION:turnN:nodeM is
  valid only if the turn is not in the future and the node is not SUPERSEDED.

  Rule §5.5.5 — BROADCAST scope: in emergency conditions (priority >= 0.95,
  high-weight cross-session references), Layer 3 can disambiguate the scope of
  an ambiguous BROADCAST. The receiving agent responds with EXCEPTION_ACCEPT.

  Rule §5.5.6 — SYNC_BARRIER: a SYNC_BARRIER node suspends the linked action
  until all listed sessions have issued a COMMIT on the synchronized action.
  When the timeout expires, TIMEOUT_ESCALATE is emitted.

  Rule §5.5.9 — Quality-constraint: a CONSTRAINT with an explicit
  quality_threshold is not satisfied if the edges enabling the constrained
  action have a quality below the threshold, even at high weight.

PYTHON IMPLEMENTATION

  The library is split into three modules in the rlp4ai/ folder:

  model.py   — data structures: all enums (RoleType, NodeType, NodeStatus,
               RelType, SpeechAct, UncertaintyType, Compression, Modality),
               dataclasses (Node, Edge, RoleBinding, StateTensor, SemanticGraph,
               InterfaceLayer, Message, CrossRef, SpeechActSet).

  engine.py  — semantic engine: resolve_concurrent_edges (rule §5.5.1),
               check_constraint_quality (rule §5.5.9), check_condition_ref
               (rules §5.5.2 and §5.5.8), should_accept_broadcast (rule §5.5.5),
               MessageValidator (full message validation).

  session.py — session store and history management: SessionStore (message
               ingestion with Event Sourcing, cross-turn reference resolution,
               SYNC_BARRIER updates), SyncBarrierState (barrier state, COMMIT
               registration, timeout extensions).


================================================================================
3. PLAI — WHAT IT IS AND HOW IT WORKS
================================================================================

PLAI (Prompting Language AI) is a declarative language that elevates Layer 4 of
RLP4AI into a complete language. It is the human-readable frontier of the
framework.

A PLAI PROGRAM HAS THIS STRUCTURE:

  @SENDER [ROLE] -> TARGET | SESSION [| priority=X] [| ttl=Xms]

  ACT_TYPE [+ ACT_TYPE ...] {
    CONFIDENCE X [| uncertainty=TYPE]

    NODE_TYPE "value"
      [salience = X]
      [status = STATUS]
      [authority = ROLE]
      [quality_min = X]
      [-> REL_TYPE "target" [weight=X] [quality=X] [if CONDITION]]
  }

  NOTE: "Natural language text."   (optional)

THE HEADER defines sender (with @), role (in []), target (after ->), session
(after |), and optional priority and TTL parameters.

THE ACT BLOCK defines the type of communicative act and the semantic content.
Acts can be composed with + (conjunction of effects): WARN+DENY means "active
warning AND action blocked". WARN alone does not imply blocking.

NODES are the semantic entities of the message. Each node has a type, a string
value, and optional attributes. Edges are declared inline with ->.

COMMENTS start with # and are ignored by the compiler.

THE NOTE at the end becomes the nl_summary field of Layer 4.

COMPILER (plai/compiler.py)

  The compiler follows three phases:
  1. Lexical tokenization (lexer): transforms the source text into a sequence of
     typed tokens (keywords, strings, symbols, numbers).
  2. Parsing: the parser builds the RLP4AI data structures (nodes, edges, header,
     acts) from the tokens.
  3. Resolution and validation: node name references are resolved to integer
     node_ids; the MessageValidator checks all operational semantic rules.
     Errors block compilation (CompileError); non-blocking violations are
     returned as warnings.

  Main function: compile_plai(source: str) -> (Message, list[str])

DECOMPILER (plai/decompiler.py)

  The decompiler reconstructs readable PLAI source from an RLP4AI Message object.
  With annotated=True it adds inline comments for each relevant semantic rule
  (blocks, violated constraints, conditional edges) and a final semantic analysis.
  It also supports decompilation from raw JSON via Decompiler.from_json().

CLI (plai/cli.py)

  The interactive CLI exposes the following commands:
    compile    — compile PLAI source to RLP4AI JSON
    decompile  — convert RLP4AI JSON to annotated PLAI
    run        — compile + validate + show JSON and decompiled PLAI
    multiline  — enter PLAI source over multiple lines (end with END)
    example    — show and run one of the 5 preloaded examples
    history    — show message history for the current session
    last       — re-display the last message (PLAI + JSON)
    annotated  — toggle semantic annotations in the decompiler
    help       — show available commands
    quit       — exit

  Shortcuts: pasting PLAI source directly (starting with @) automatically
  triggers compilation; pasting RLP4AI JSON (starting with {) automatically
  triggers decompilation.


================================================================================
4. REPOSITORY STRUCTURE
================================================================================

rlp4ai_and_plai/
├── rlp4ai/
│   ├── __init__.py
│   ├── model.py          protocol data structures
│   ├── engine.py         semantic engine and validation
│   └── session.py        session store, Event Sourcing, SYNC_BARRIER
├── plai/
│   ├── __init__.py
│   ├── compiler.py       PLAI -> RLP4AI compiler
│   ├── decompiler.py     RLP4AI -> PLAI decompiler
│   └── cli.py            command-line interface
└── tests/
    ├── __init__.py
    ├── test_rlp4ai.py    58 automated tests for the RLP4AI library
    └── test_plai.py      42 automated tests for PLAI


================================================================================
5. HOW TO USE RLP4AI (PYTHON EXAMPLES)
================================================================================

Requirements: Python 3.10 or higher. No external dependencies.

All imports must be run from the repository root (the folder containing rlp4ai/
and plai/).

Example 1 — build a graph and apply the concurrent weight rule:

    from rlp4ai.model import Node, Edge, SemanticGraph, NodeType, RelType, SpeechActSet
    from rlp4ai.engine import resolve_concurrent_edges

    nodes = [
        Node(0, NodeType.ACTION, "analyze_dataset"),
        Node(1, NodeType.FAILED_STATE, "preprocessing_failed"),
        Node(2, NodeType.STATE, "raw_data_available"),
    ]
    edges = [
        Edge(from_id=1, to_id=0, rel_type=RelType.BLOCKS, weight=0.85),
        Edge(from_id=2, to_id=0, rel_type=RelType.ENABLES, weight=0.60),
    ]
    graph = SemanticGraph(nodes=nodes, edges=edges,
                          speech_act=SpeechActSet.parse("WARN+PROPOSE_ALTERNATIVE"))

    result = resolve_concurrent_edges(graph, target_node_id=0)
    print(result.description)
    # -> "BLOCKED: weight_BLOCKS(0.85) > weight_ENABLES(0.60). WARN or DENY required."

Example 2 — validate a complete message:

    from rlp4ai.model import (
        RoleBinding, StateTensor, SemanticGraph, Message,
        RoleType, UncertaintyType, SpeechActSet
    )
    from rlp4ai.engine import MessageValidator

    rb = RoleBinding(
        sender_id="AGENT-A", role_type=RoleType.COORDINATOR,
        target_id="AGENT-B", session_id="SESSION-001",
        turn_idx=1, priority=0.9,
    )
    st = StateTensor(confidence=0.9, uncertainty_type=UncertaintyType.NONE)
    sg = SemanticGraph(speech_act=SpeechActSet.parse("DELEGATE"))
    msg = Message(rb, st, sg)

    result = MessageValidator().validate(msg)
    print(result)
    # -> ValidationResult(OK)

Example 3 — check a composed speech_act:

    from rlp4ai.model import SpeechActSet, SpeechAct

    sa = SpeechActSet.parse("WARN+DENY")
    print(sa.blocks())    # True
    print(sa.warns())     # True
    print(sa.commits())   # False

Example 4 — use the SessionStore with Event Sourcing:

    from rlp4ai.session import SessionStore
    # (build msg1 and msg2 as above, with turn_idx=1 and turn_idx=2)
    store = SessionStore()
    store.ingest(msg1)
    store.ingest(msg2)
    # retrieve a node from history
    from rlp4ai.model import CrossRef
    node = store.get_node(CrossRef.parse("@SESSION-001:turn1:node0"))
    print(node)


================================================================================
6. HOW TO USE PLAI
================================================================================

STARTING THE INTERACTIVE CLI

    cd /path/to/rlp4ai_and_plai
    python3 plai/cli.py

    # The prompt is:
    plai>

    # To see the 5 preloaded examples:
    plai> example

    # To run example 1 (delegation with deadline):
    plai> example 1

    # To compile PLAI source pasted directly:
    plai> @AGENT-A [COORDINATOR] -> AGENT-B | SESSION-042
    # (compilation starts automatically when the line begins with @)

    # For multiline mode:
    plai> multiline
    # (paste the source, end with END)

    # To toggle semantic annotations:
    plai> annotated

    # To exit:
    plai> quit

COMPILING A .plai FILE

    python3 plai/cli.py my_message.plai
    # prints the RLP4AI JSON to stdout

    python3 plai/cli.py my_message.plai > output.json
    # saves the JSON to a file

USING PLAI FROM PYTHON

    Compile PLAI source:

        from plai.compiler import compile_plai, CompileError

        source = '''
        @AGENT-A [COORDINATOR] -> AGENT-B | SESSION-001

        DELEGATE {
          CONFIDENCE 0.90
          GOAL "deliver_report_X"
          CONSTRAINT "deadline=T+20min"
            authority = COORDINATOR
        }
        '''
        try:
            msg, warnings = compile_plai(source)
            print(f"Nodes: {len(msg.semantic_graph.nodes)}")
            for w in warnings:
                print(f"Warning: {w}")
        except CompileError as e:
            print(f"Error: {e}")

    Decompile an RLP4AI message to PLAI:

        from plai.decompiler import Decompiler

        d = Decompiler()
        plai_src = d.decompile(msg, annotated=True)
        print(plai_src)

    Decompile from raw JSON:

        from plai.decompiler import Decompiler

        msg, err = Decompiler.from_json(raw_json_string)
        if not err:
            print(Decompiler().decompile(msg, annotated=True))


================================================================================
7. HOW TO RUN THE TESTS
================================================================================

INSTALLATION (once only)

    pip install pytest

    If pip is not on the PATH:
        python3 -m pip install pytest

    The library has no other external dependencies: it uses only standard Python
    modules (dataclasses, enum, hashlib, json, re, uuid, time).

RUN ALL TESTS

    cd /path/to/rlp4ai_and_plai
    python3 -m pytest tests/ -v

    Expected output: 100 passed (58 RLP4AI tests + 42 PLAI tests)

RUN ONLY THE RLP4AI TESTS

    python3 -m pytest tests/test_rlp4ai.py -v

RUN ONLY THE PLAI TESTS

    python3 -m pytest tests/test_plai.py -v

RUN A SPECIFIC GROUP OF TESTS

    python3 -m pytest tests/ -v -k "TestSyncBarrier"
    python3 -m pytest tests/ -v -k "TestCompiler"
    python3 -m pytest tests/ -v -k "TestLexer"

RUN ONLY FAILED TESTS (useful for debugging)

    python3 -m pytest tests/ -v --tb=short

NOTE: tests must be run from the repository root (the folder containing rlp4ai/
and plai/), otherwise Python cannot find the modules and returns
ModuleNotFoundError.


================================================================================
8. DEPENDENCIES
================================================================================

Python 3.10 or higher.

Runtime dependencies: none. The code uses only standard library modules.
  dataclasses, enum, hashlib, json, re, uuid, time, collections

Development dependencies:
  pytest >= 7.0    (only required to run the tests)

Installation:
  pip install pytest

================================================================================
