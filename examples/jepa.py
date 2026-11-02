"""Run a tiny, synthetic JEPA-style training loop through canonical graphs."""
import argparse
import importlib.metadata
import json
import math
import random
from pathlib import Path

from tinygrad import Device, Tensor
from tinygrad.helpers import Context
from tinygrad.nn.optim import SGD

from tgaoi.aoi import ema_update_graph, jepa_graph
from tgaoi.backends import TinygradBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--graph', type=Path, help='optional canonical JSON export')
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 2:
        parser.error('steps must be positive and batch-size must be at least two')
    rng = random.Random(args.seed)

    def normal(shape, scale=1.0):
        # Host initialization keeps the tiny demo cheap on interpreter devices.
        return Tensor([rng.gauss(0, scale) for _ in range(math.prod(shape))]).reshape(shape).realize()

    graph, ema = jepa_graph(), ema_update_graph()
    backend = TinygradBackend()
    state = {}
    for key, typ in graph.inputs.items():
        if key in ('context_view', 'target_view') or key.startswith('target_encoder.'):
            continue
        state[key] = normal(typ.shape, 0.1).detach()
    for key in list(state):
        if key.startswith('context_encoder.'):
            target_key = key.replace('context_encoder.', 'target_encoder.', 1)
            state[target_key] = state[key].detach().clone().realize()
    trainable = [v for k, v in state.items() if not k.startswith('target_encoder.')]
    optimizer = SGD(trainable, lr=0.05, fused=False)
    # Fixed synthetic paired views: context hides half the input coordinates.
    # No labels, downloads, pretrained weights or representation-quality claim.
    target_view = normal((args.batch_size, 8))
    context_view = (target_view * Tensor([1., 1., 1., 1., 0., 0., 0., 0.])).realize()
    feeds = {**state, 'context_view': context_view, 'target_view': target_view}
    losses = []
    with Context(TRAINING=1):
        for _ in range(args.steps):
            optimizer.zero_grad()
            outputs = backend.run(graph, feeds)
            loss = outputs['loss']
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError('nonfinite JEPA loss')
            losses.append(value)
            loss.backward()
            optimizer.step()
            for key in state:
                if key.startswith('target_encoder.'):
                    online_key = key.replace('target_encoder.', 'context_encoder.', 1)
                    next_target = backend.run(ema, {'target': state[key], 'online': state[online_key]})['target_next']
                    state[key].assign(next_target).realize()
    if args.graph:
        args.graph.parent.mkdir(parents=True, exist_ok=True)
        args.graph.write_text(json.dumps(graph.to_dict(), indent=2) + '\n')
    report = {'example': 'synthetic_vector_jepa', 'seed': args.seed, 'steps': args.steps,
              'batch_size': args.batch_size, 'device': Device.DEFAULT,
              'tinygrad_version': importlib.metadata.version('tinygrad'),
              'optimizer': 'SGD', 'learning_rate': 0.05, 'ema_decay': ema.metadata['decay'],
              'initial_loss': losses[0], 'last_pre_update_loss': losses[-1],
              'note': 'execution smoke test; loss decrease does not establish useful representations'}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
