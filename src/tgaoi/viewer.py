"""Export an offline, read-only viewer from model artifacts, without model imports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .audit import catalog, audit_graph
from .ir import Graph


def load_package(package: Path):
    import yaml
    architecture = yaml.safe_load((package / "architecture_map.yaml").read_text())
    dictionary = yaml.safe_load((package / "model_dictionary.yaml").read_text())
    if not isinstance(architecture, dict) or not isinstance(dictionary, dict):
        raise ValueError("Map and dictionary must be mappings")
    definitions = catalog()
    blocks = {}
    for section in ("blocks", "training_blocks"):
        for key, block in dictionary.get(section, {}).items():
            if "aoi" in block:
                blocks[block["aoi"]] = {**block, "dictionary_key": key, "section": section}
    nodes = []
    seen = set()
    for section in ("nodes", "training"):
        for entry in architecture.get(section, []):
            ident = entry.get("canonical_id", entry.get("canonical_id_pattern"))
            if not ident or ident in seen:
                raise ValueError(f"Missing/duplicate canonical ID: {ident}")
            seen.add(ident)
            nodes.append({"source_file": dictionary.get("model", {}).get(
                "training_entrypoint" if section == "training" else "source_entrypoint"),
                **entry, "id": ident, "section": section})
    # Parentage is a display projection of canonical ID prefixes, not dataflow.
    for node in nodes:
        parents = [n["id"] for n in nodes if node["id"].startswith(n["id"] + ".")]
        node["parent"] = max(parents, key=len) if parents else None
    return {"schema_version": 1, "model": architecture["model"], "nodes": nodes,
            "blocks": blocks, "parameter_materialization": architecture.get("parameter_materialization", {}),
            "notes": dictionary.get("notes", []),
            "catalog": {name: {"graph": g.to_dict(), "audit": audit_graph(g, definitions)}
                        for name, g in definitions.items()}, "graphs": []}


def render(data):
    # Escape HTML parser delimiters; all dynamic UI text uses textContent.
    payload = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return Path(__file__).with_name("viewer.html").read_text().replace("__VIEWER_DATA__", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("reference_models/rfdetr"))
    parser.add_argument("--graph", action="append", type=Path, default=[], help="Canonical graph JSON; repeat for multiple graphs")
    parser.add_argument("--output", type=Path, default=Path("artifacts/viewer.html"))
    args = parser.parse_args(argv)
    data = load_package(args.package)
    for path in args.graph:
        graph = Graph.from_dict(json.loads(path.read_text()))
        data["graphs"].append({"graph": graph.to_dict(), "audit": audit_graph(graph)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
