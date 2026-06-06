from __future__ import annotations

import sys
import os
import json
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlp4ai.engine import MessageValidator
from plai.compiler import compile_plai, CompileError, LexError, ParseError
from plai.decompiler import Decompiler


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"

    ok    = staticmethod(lambda s: f"\033[92m{s}\033[0m")
    warn  = staticmethod(lambda s: f"\033[93m{s}\033[0m")
    err   = staticmethod(lambda s: f"\033[91m{s}\033[0m")
    info  = staticmethod(lambda s: f"\033[96m{s}\033[0m")
    bold  = staticmethod(lambda s: f"\033[1m{s}\033[0m")
    dim   = staticmethod(lambda s: f"\033[2m{s}\033[0m")
    title = staticmethod(lambda s: f"\033[1m\033[95m{s}\033[0m")



def _msg_to_dict(msg) -> dict:
    rb = msg.role_binding
    st = msg.state_tensor
    sg = msg.semantic_graph

    nodes_list = []
    for n in sg.nodes:
        nd = {
            "node_id": n.node_id,
            "type": n.type.value,
            "value": n.value,
            "salience": round(n.salience, 2),
            "status": n.status.value,
        }
        if n.authority_required:
            nd["authority_required"] = n.authority_required
        if n.quality_threshold != 1.0:
            nd["quality_threshold"] = round(n.quality_threshold, 2)
        if n.type.value == "SYNC_BARRIER" and n.sync_sessions:
            nd["sync_sessions"] = n.sync_sessions
            nd["sync_timeout_ms"] = n.sync_timeout_ms
            nd["sync_action"] = n.sync_action
        nodes_list.append(nd)

    edges_list = []
    for e in sg.edges:
        ed = {
            "from": e.from_id,
            "to": str(e.to_id),
            "rel_type": e.rel_type.value,
            "weight": round(e.weight, 2),
            "quality": round(e.quality, 2),
        }
        if e.condition_ref:
            ed["condition_ref"] = e.condition_ref
        edges_list.append(ed)

    result = {
        "LAYER_1": {
            "sender_id": rb.sender_id,
            "role_type": rb.role_type.value,
            "target_id": rb.target_id,
            "session_id": rb.session_id,
            "turn_idx": rb.turn_idx,
            "ttl_ms": rb.ttl_ms,
            "priority": round(rb.priority, 2),
        },
        "LAYER_2": {
            "confidence": round(st.confidence, 2),
            "uncertainty_type": st.uncertainty_type.value,
            "delta_flag": st.delta_flag,
        },
        "LAYER_3": {
            "nodes": nodes_list,
            "edges": edges_list,
            "speech_act": str(sg.speech_act),
            "intent_vector": sg.intent_description,
        },
    }
    if msg.interface and msg.interface.nl_summary:
        result["LAYER_4"] = {"nl_summary": msg.interface.nl_summary}
    return result



EXAMPLES = {
    "1": {
        "title": "Delega con deadline e formato",
        "source": """\
@AGENT-A [COORDINATOR] → AGENT-B | SESSION-042 | priority=0.90

DELEGATE {
  CONFIDENCE 0.90

  GOAL "deliver_report_X"
    salience = 1.0

  ACTION "analyze_dataset_D"
    salience = 0.95
    → ENABLES "deliver_report_X" weight=0.9

  ACTION "generate_summary"
    salience = 0.90
    → ENABLES "deliver_report_X" weight=0.95

  CONSTRAINT "deadline=T+20min"
    salience = 1.0
    authority = COORDINATOR
    → REQUIRES "deliver_report_X" weight=1.0

  CONSTRAINT "output_format=JSON"
    salience = 0.75
    → REQUIRES "generate_summary" weight=0.75
}

NOTE: "Analizza il dataset D, genera il summary e consegna il report X in JSON entro 20 minuti."
""",
    },
    "2": {
        "title": "Avviso con FAILED_STATE e proposta alternativa",
        "source": """\
@AGENT-B [EXECUTOR] → AGENT-A | SESSION-042 | priority=0.93

WARN+PROPOSE_ALTERNATIVE {
  CONFIDENCE 0.87 | uncertainty=EPISTEMIC

  FAILED_STATE "preprocessing_step_P_failed"
    salience = 0.95
    → BLOCKS "analyze_dataset_D" weight=0.85

  STATE "raw_data_available"
    salience = 0.70
    → ENABLES "analyze_dataset_D" weight=0.60

  RISK "high_error_rate_on_raw_data"
    salience = 0.85

  ACTION "execute_fallback_preprocessing"
    salience = 0.90
    → RESOLVES "preprocessing_step_P_failed" weight=0.90 quality=0.85
    → MITIGATES "high_error_rate_on_raw_data" weight=0.80

  CONSTRAINT "deadline_extended=T+25min"
    authority = COORDINATOR
    salience = 0.80
}

NOTE: "Il preprocessing P è fallito. Propongo fallback + estensione deadline di 5 minuti."
""",
    },
    "3": {
        "title": "Impegno condizionale con SYNC_BARRIER",
        "source": """\
@AGENT-D [CLUSTER_MANAGER] → AGENT-A | SESSION-303 | priority=0.95

WARN+COMMIT {
  CONFIDENCE 0.85 | uncertainty=ALEATORIC

  FAILED_STATE "integrity_check_table_orders_failed"
    salience = 0.95

  ACTION "execute_auto_repair"
    salience = 1.0
    → RESOLVES "integrity_check_table_orders_failed" weight=0.90 quality=0.90

  RISK "auto_repair_time_variance"
    salience = 0.80
    → DEGRADES "execute_auto_repair" weight=0.30

  SYNC_BARRIER "pre_cutover_sync"
    salience = 1.0
    sessions = [SESSION-301 | SESSION-302 | SESSION-303]
    timeout = 47000
    sync_action = "integrity_check_COMPLETE"
}

NOTE: "Avvio riparazione automatica. Margine 12s sul barrier. Rischio varianza I/O dichiarato."
""",
    },
    "4": {
        "title": "Override con OVERRIDE_GRANT",
        "source": """\
@AGENT-A [INCIDENT_COMMANDER] → BROADCAST | SESSION-GLOBAL | priority=1.0

OVERRIDE_GRANT+DELEGATE {
  CONFIDENCE 0.91 | uncertainty=ALEATORIC

  ACTION "execute_full_isolation_BOTH_clusters"
    salience = 1.0

  CONSTRAINT "isolation_must_complete_within_5min"
    salience = 1.0
    authority = INCIDENT_COMMANDER
    → REQUIRES "execute_full_isolation_BOTH_clusters" weight=1.0

  STATE "breach_lateral_movement_confirmed"
    salience = 0.95
    → CAUSES "execute_full_isolation_BOTH_clusters" weight=0.90
}

NOTE: "Ordine immediato: isolamento totale entrambi i cluster. Override su SLA autorizzato."
""",
    },
    "5": {
        "title": "Query con QUERY_NODE e timeout",
        "source": """\
@AGENT-A [COORDINATOR] → AGENT-D | SESSION-303 | priority=0.93 | ttl=15000ms

QUERY {
  CONFIDENCE 0.88 | uncertainty=ALEATORIC

  STATE "north_and_south_ready_waiting_east"
    salience = 0.95

  RISK "sync_timeout_breach_imminent"
    salience = 0.90
}

NOTE: "NORTH e SOUTH pronti. Puoi completare l'integrity check entro il timeout? Rispondi entro 15s."
""",
    },
}



COMMANDS = {
    "compile":    "Compila sorgente PLAI → RLP4AI JSON",
    "decompile":  "Converti RLP4AI JSON → sorgente PLAI",
    "run":        "Compila ed esegui validazione completa",
    "multiline":  "Inserisci sorgente PLAI su più righe (termina con END)",
    "example":    "Mostra/esegui esempi: example [1-5]",
    "session":    "Mostra/modifica parametri sessione",
    "history":    "Mostra cronologia messaggi",
    "last":       "Rimostra l'ultimo messaggio",
    "annotated":  "Toggle annotazioni semantiche nel decompilatore",
    "help":       "Mostra questa guida",
    "quit":       "Esci",
}


def print_banner() -> None:
    banner = r"""
  ██████╗ ██╗      █████╗ ██╗
  ██╔══██╗██║     ██╔══██╗██║
  ██████╔╝██║     ███████║██║
  ██╔═══╝ ██║     ██╔══██║██║
  ██║     ███████╗██║  ██║██║
  ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝
  Prompting Language AI  —  Framework PLAI + RLP4AI v1.0
"""
    print(C.title(banner))
    print(C.dim("  H2A (Human→Agent)  e  A2H (Agent→Human)\n"))
    print(C.dim("  'help' → comandi  |  'example' → esempi  |  'quit' → esci\n"))


def print_help() -> None:
    print(C.title("\n  COMANDI PLAI CLI"))
    print()
    for cmd, desc in COMMANDS.items():
        print(f"  {C.bold(cmd):<18} {desc}")
    print()
    print(C.info("  SCORCIATOIE:"))
    print("  • Incolla direttamente sorgente PLAI (inizia con @) → compila automaticamente")
    print("  • Incolla JSON RLP4AI (inizia con {) → decompila automaticamente")
    print()



class CLISession:
    def __init__(self):
        self.history: list[dict] = []
        self.annotated: bool = True
        self.decompiler = Decompiler()


def _store(session: CLISession, direction: str, source: str, msg) -> None:
    session.history.append({
        "n": len(session.history) + 1,
        "dir": direction,
        "src": source[:60],
        "msg": msg,
    })



def do_compile(session: CLISession, source: str) -> None:
    try:
        msg, warnings = compile_plai(source)
    except CompileError as e:
        print(C.err(f"\n   Errore di compilazione:\n    {e}"))
        return

    for w in warnings:
        print(C.warn(f"    {w}"))

    print(C.ok(f"\n   Compilato  ({len(msg.semantic_graph.nodes)} nodi, "
               f"{len(msg.semantic_graph.edges)} archi)  "
               f"checksum: {msg.checksum()[:12]}…"))

    d = _msg_to_dict(msg)
    print(C.info("\n  ── RLP4AI JSON ──"))
    print(json.dumps(d, indent=2, ensure_ascii=False))

    _store(session, "PLAI→RLP", source, msg)


def do_decompile(session: CLISession, raw: str) -> None:
    if not raw.strip().endswith("}"):
        print(C.dim("  Incolla il JSON (termina con END):"))
        lines = [raw] if raw.strip() else []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().upper() == "END":
                break
            lines.append(line)
        raw = "\n".join(lines)

    msg, err = Decompiler.from_json(raw)
    if err:
        print(C.err(f"\n   {err}"))
        return

    plai_src = session.decompiler.decompile(msg, annotated=session.annotated)
    print(C.info("\n  ── PLAI ──"))
    print(plai_src)

    _store(session, "RLP→PLAI", raw[:60], msg)


def do_run(session: CLISession, source: str) -> None:
    try:
        msg, warnings = compile_plai(source)
    except CompileError as e:
        print(C.err(f"\n   {e}"))
        return

    for w in warnings:
        print(C.warn(f"    {w}"))

    validator = MessageValidator()
    result = validator.validate(msg)
    if result.valid:
        print(C.ok(f"\n   Valido  |  checksum: {msg.checksum()[:12]}…"))
    else:
        print(C.err(f"\n   {len(result.errors)} errori di validazione:"))
        for e in result.errors:
            print(C.err(f"    • {e}"))
        return

    print(C.info("\n  ── RLP4AI JSON ──"))
    print(json.dumps(_msg_to_dict(msg), indent=2, ensure_ascii=False))

    print(C.info("\n  ── Decompilato in PLAI ──"))
    print(session.decompiler.decompile(msg, annotated=session.annotated))

    _store(session, "PLAI→RLP", source, msg)


def do_multiline(session: CLISession) -> None:
    print(C.dim("  Inserisci sorgente PLAI. Termina con una riga che contiene solo END:"))
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    source = "\n".join(lines)
    if source.strip():
        do_run(session, source)
    else:
        print(C.dim("  Nessun input."))


def do_example(session: CLISession, arg: str) -> None:
    num = arg.strip()
    if not num:
        print(C.info("\n  Esempi disponibili:"))
        for k, v in EXAMPLES.items():
            print(f"  {C.bold(k)}.  {v['title']}")
        print(C.dim("\n  Usa: example <numero>  per eseguirlo"))
        return

    ex = EXAMPLES.get(num)
    if not ex:
        print(C.warn(f"  Esempio '{num}' non trovato. Disponibili: {', '.join(EXAMPLES.keys())}"))
        return

    print(C.title(f"\n  ESEMPIO {num}: {ex['title']}"))
    print(C.info("\n  ── Sorgente PLAI ──"))
    print(ex["source"])
    do_run(session, ex["source"])


def do_history(session: CLISession) -> None:
    if not session.history:
        print(C.dim("  Cronologia vuota."))
        return
    print(C.info(f"\n  Cronologia ({len(session.history)} messaggi):"))
    for item in session.history:
        print(f"  [{item['n']}] {C.bold(item['dir'])}  {C.dim(item['src'])}")


def do_last(session: CLISession) -> None:
    if not session.history:
        print(C.dim("  Nessun messaggio ancora."))
        return
    last = session.history[-1]
    msg = last["msg"]
    print(C.info(f"\n  Ultimo [{last['n']}] {last['dir']}:"))
    print(C.info("\n  ── PLAI (decompilato) ──"))
    print(session.decompiler.decompile(msg, annotated=session.annotated))
    print(C.info("\n  ── RLP4AI JSON ──"))
    print(json.dumps(_msg_to_dict(msg), indent=2, ensure_ascii=False))


def do_annotated(session: CLISession) -> None:
    session.annotated = not session.annotated
    state = "ON" if session.annotated else "OFF"
    print(C.ok(f"  Annotazioni semantiche: {state}"))



def run_cli() -> None:
    print_banner()
    session = CLISession()

    while True:
        try:
            raw = input(f"{C.BOLD}{C.MAGENTA}plai{C.RESET}{C.DIM}>{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(C.dim("\n\n  Arrivederci."))
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print(C.dim("\n  Arrivederci."))
            break
        elif cmd == "help":
            print_help()
        elif cmd in ("compile", "c"):
            if arg:
                do_compile(session, arg)
            else:
                do_multiline(session)
        elif cmd in ("decompile", "d"):
            do_decompile(session, arg)
        elif cmd in ("run", "r"):
            if arg:
                do_run(session, arg)
            else:
                do_multiline(session)
        elif cmd in ("multiline", "m", "ml"):
            do_multiline(session)
        elif cmd in ("example", "ex"):
            do_example(session, arg)
        elif cmd == "history":
            do_history(session)
        elif cmd == "last":
            do_last(session)
        elif cmd == "annotated":
            do_annotated(session)
        elif raw.startswith("@"):
            do_run(session, raw)
        elif raw.startswith("{"):
            do_decompile(session, raw)
        else:
            print(C.warn(f"  Comando sconosciuto: '{cmd}'. Digita 'help'."))

        print()



def run_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except FileNotFoundError:
        print(f"File non trovato: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        msg, warnings = compile_plai(source)
    except CompileError as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)

    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)

    print(json.dumps(_msg_to_dict(msg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--help", "-h"):
            print(__doc__)
            print_help()
        else:
            run_file(sys.argv[1])
    else:
        run_cli()
