"""Revision-pinned, non-executing Tinygrad source-to-AOI inventory.

This is a source projection, never an executable canonical graph. Unknown or
ambiguous calls remain opaque. Method-name hints are not verified mappings.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import gzip
import json
import subprocess
from collections import Counter
from pathlib import Path

SCHEMA = 1
RULES_VERSION = 2
LIBRARY_BLOCKS = {
    ('examples/yolov8.py', 'Conv_Block'): 'YOLOV8_CONV',
    ('examples/yolov8.py', 'Bottleneck'): 'YOLOV8_BOTTLENECK',
    ('examples/yolov8.py', 'C2f'): 'YOLOV8_C2F',
    ('examples/yolov8.py', 'SPPF'): 'YOLOV8_SPPF',
    ('tinygrad/llm/model.py', 'precompute_freqs_cis'): ['ROTARY_FREQUENCIES'],
    ('tinygrad/llm/model.py', 'apply_rope'): ['ROPE_HALF_SPLIT'],
    ('tinygrad/llm/model.py', 'FFNBlock._feed_forward'): ['SWIGLU'],
    ('tinygrad/llm/model.py', 'TransformerBlock._attention'): ['MASKED_GQA', 'KV_APPEND', 'ROPE_HALF_SPLIT'],
    ('tinygrad/llm/model.py', 'GatedDeltaNetBlock._attention'): [
        'CAUSAL_DEPTHWISE_CONV_STATE','L2_NORMALIZE','DELTA_GATES','GATED_DELTA_STEP','GATED_DELTA_SCAN','GATED_RMSNORM'],
}
API_AOIS = {
    'tinygrad.nn.Linear': 'LINEAR', 'tinygrad.nn.Conv2d': 'CONV2D',
    'tinygrad.nn.Conv1d': 'CONV1D', 'tinygrad.nn.ConvTranspose2d': 'CONV_TRANSPOSE2D',
    'tinygrad.nn.BatchNorm': 'BATCHNORM', 'tinygrad.nn.BatchNorm2d': 'BATCHNORM',
    'tinygrad.nn.LayerNorm': 'LAYERNORM', 'tinygrad.nn.LayerNorm2d': 'NORM2D',
    'tinygrad.nn.RMSNorm': 'RMSNORM', 'tinygrad.nn.GroupNorm': 'GROUPNORM',
    'tinygrad.nn.InstanceNorm': 'INSTANCENORM', 'tinygrad.nn.Embedding': 'EMBEDDING',
    'tinygrad.nn.optim.Adam': 'ADAM_BIAS_CORRECTED',
    'tinygrad.nn.optim.AdamW': 'ADAMW_STEP', 'tinygrad.nn.optim.SGD': 'SGD_STEP',
    'tinygrad.nn.optim.Muon': 'MUON_STEP',
}
METHOD_AOIS = {
    'linear': 'LINEAR', 'matmul': 'MATMUL', 'dot': 'MATMUL', 'conv2d': 'CONV2D',
    'softmax': 'SOFTMAX', 'log_softmax': 'LOG_SOFTMAX', 'layernorm': 'LAYERNORM',
    'scaled_dot_product_attention': 'ATTENTION', 'relu': 'ACTIVATION::RELU',
    'gelu': 'ACTIVATION::GELU', 'silu': 'ACTIVATION::SILU', 'sigmoid': 'SIGMOID',
    'tanh': 'TANH', 'dropout': 'DROPOUT', 'reshape': 'RESHAPE', 'permute': 'PERMUTE',
    'transpose': 'TRANSPOSE', 'flatten': 'FLATTEN', 'cat': 'CONCAT', 'stack': 'STACK',
    'split': 'SPLIT', 'chunk': 'SPLIT', 'pad': 'PAD', 'gather': 'GATHER',
    'max_pool2d': 'MAX_POOL2D', 'avg_pool2d': 'AVG_POOL2D', 'interpolate': 'INTERPOLATE',
    'mean': 'REDUCE_MEAN', 'sum': 'REDUCE_SUM', 'max': 'REDUCE_MAX', 'exp': 'EXP',
    'sqrt': 'SQRT', 'rsqrt': 'RSQRT', 'sequential': 'SEQUENTIAL',
    'sparse_categorical_crossentropy': 'CROSS_ENTROPY', 'backward': 'BACKWARD',
}
for _module in ('tinygrad.Tensor', 'tinygrad.tensor.Tensor'):
    API_AOIS.update({f'{_module}.{k}': v for k, v in METHOD_AOIS.items()})


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f'{base}.{node.attr}' if base else None
    return None


def module_name(path):
    name = path[:-3].replace('/', '.')
    return name.removesuffix('.__init__')


def symbol_id(module, name):
    return f'SOURCE::{module}::{name}'


class Projection(ast.NodeVisitor):
    def __init__(self, module, tree):
        self.module = module
        self.scopes = []
        self.records = []
        self.aliases = {}
        # Only unconditional top-level imports are treated as resolved APIs.
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    if alias.name != '*':
                        self.aliases[alias.asname or alias.name] = f'{node.module}.{alias.name}'
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        # Reassignments/shadowing anywhere conservatively invalidate an import.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.aliases.pop(node.id, None)
            elif isinstance(node, ast.arg):
                self.aliases.pop(node.arg, None)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.aliases.pop(node.name, None)
        self.records.append({'id': symbol_id(module, '<module>'), 'name': '<module>',
                             'kind': 'module', 'parent': None, 'line': 1, 'end_line': 1,
                             'calls': [], 'signature': None})
        self.current = self.records[0]
        self.shadowed = set()

    def definition(self, node, kind):
        name = '.'.join([*self.scopes, node.name])
        record = {'id': symbol_id(self.module, name), 'name': name, 'kind': kind,
                  'parent': self.current['id'], 'line': node.lineno, 'end_line': node.end_lineno,
                  'calls': [], 'signature': ast.unparse(node.args) if hasattr(node, 'args') else None}
        self.records.append(record)
        previous_shadowed = self.shadowed
        self.shadowed = self.shadowed | {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if hasattr(node, 'args'):
            self.shadowed |= {n.arg for n in ast.walk(node.args) if isinstance(n, ast.arg)}
        previous = self.current
        self.current = record
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()
        self.current = previous
        self.shadowed = previous_shadowed

    def visit_ClassDef(self, node):
        self.definition(node, 'class')

    def visit_FunctionDef(self, node):
        self.definition(node, 'function')

    def visit_AsyncFunctionDef(self, node):
        self.definition(node, 'async_function')

    def visit_Call(self, node):
        raw = dotted(node.func)
        qualified = None
        if raw:
            head, *tail = raw.split('.')
            if head in self.aliases:
                qualified = '.'.join([self.aliases[head], *tail])
        mapped = API_AOIS.get(qualified)
        hint = METHOD_AOIS.get(node.func.attr) if isinstance(node.func, ast.Attribute) else None
        self.current['calls'].append({
            'id': f'{self.current["id"]}@{node.lineno}:{node.col_offset}',
            'line': node.lineno, 'column': node.col_offset,
            'expression': ast.unparse(node.func),
            'arguments': [ast.unparse(a) for a in node.args],
            'keywords': {k.arg or '**': ast.unparse(k.value) for k in node.keywords},
            'resolved_api': qualified, 'aoi': mapped,
            'candidate_aoi': hint if not mapped else None,
            'status': 'api_mapping' if mapped else 'opaque',
            'target': None, 'local_shadowed': raw in self.shadowed,
        })
        self.generic_visit(node)


def parse_source(path, source):
    tree = ast.parse(source, filename=path)
    visitor = Projection(module_name(path), tree)
    visitor.visit(tree)
    return visitor.records


def build_catalog(root: Path, revision='HEAD'):
    sha = git(root, 'rev-parse', '--verify', revision + '^{commit}').decode().strip()
    paths = git(root, 'ls-tree', '-r', '--name-only', sha, '--', 'examples', 'extra/models', 'tinygrad/llm').decode().splitlines()
    files = []
    for path in sorted(paths):
        raw = git(root, 'show', f'{sha}:{path}')
        item = {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                'source_url': f'https://github.com/tinygrad/tinygrad/blob/{sha}/{path}',
                'kind': 'python' if path.endswith('.py') else 'support_asset', 'symbols': []}
        if item['kind'] == 'python':
            try:
                item['symbols'] = parse_source(path, raw)
                item['status'] = 'source_mapped'
            except (SyntaxError, UnicodeError) as exc:
                item.update(status='parse_error', error=str(exc))
        else:
            item['status'] = 'inventoried_only'
        files.append(item)
    # Link unambiguous local/imported source declarations without importing them.
    definitions = {f'{module_name(f["path"])}.{s["name"]}': s['id']
                   for f in files for s in f['symbols'] if s['kind'] != 'module'}
    for f in files:
        for symbol in f['symbols']:
            if (mapped := LIBRARY_BLOCKS.get((f['path'], symbol['name']))) is not None:
                if isinstance(mapped, list):
                    symbol['library_aois'] = mapped
                    symbol['library_scope'] = 'Reusable float32 mathematical subgraphs only. Explicit state; no quantized weights, fused AMD kernels, full model or MoE equivalence. FFN link covers dense branch only. See QWEN_BLOCKS.md.'
                else:
                    symbol['library_aoi'] = mapped
                    symbol['library_scope'] = 'Inference only, explicit frozen BatchNorm state; tested small configurations. Bottleneck uses 3x3 kernels. Not a complete YOLOv8 translation.'
            for call in symbol['calls']:
                target = definitions.get(call['resolved_api'])
                if not target and not call['local_shadowed'] and call['expression'].isidentifier():
                    # Only top-level definitions; dynamic/local shadowing is unresolved.
                    target = definitions.get(f'{module_name(f["path"])}.{call["expression"]}')
                if target:
                    call.update(target=target, status='source_reference', aoi=None)
    counts = Counter(c['status'] for f in files for s in f['symbols'] for c in s['calls'])
    return {'schema_version': SCHEMA, 'rules_version': RULES_VERSION,
            'upstream': {'repository': 'https://github.com/tinygrad/tinygrad', 'revision': sha},
            'scope': ['examples/**', 'extra/models/**', 'tinygrad/llm/**'],
            'representation': 'static_source_projection_not_executable_ir',
            'summary': {'files': len(files), 'python_files': sum(f['kind'] == 'python' for f in files),
                        'parse_errors': sum(f['status'] == 'parse_error' for f in files),
                        'symbols': sum(len(f['symbols']) for f in files), 'calls': dict(sorted(counts.items()))},
            'files': files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--output', type=Path, default=Path('reference_models/tinygrad/catalog.json.gz'))
    args = parser.parse_args(argv)
    data = build_catalog(args.checkout, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n').encode()
    args.output.write_bytes(gzip.compress(raw, mtime=0) if args.output.suffix == '.gz' else raw)
    print(json.dumps(data['summary'], indent=2))
    if data['summary']['parse_errors']:
        raise SystemExit('Catalog contains parse errors; see per-file diagnostics')


if __name__ == '__main__':
    main()
