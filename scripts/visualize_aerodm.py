"""Trace AeroDM's denoising forward pass with torchview (no checkpoint needed).

Setup: pip install torchview graphviz; brew install graphviz
Run: python scripts/visualize_aerodm.py [--obstacles] [--depth 3]
The graph excludes the reverse-diffusion loop, CBF guidance and training loss.
"""
import argparse
from pathlib import Path
import sys
import torch
from torch import nn
from torchview import draw_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from AeroDM_ObstacleAware_Training_Test import AeroDM, Config


class TraceInputs(nn.Module):
    """Adapt obstacle tensors to the model's list-of-dictionaries interface."""
    def __init__(self, model, use_obstacles=False):
        super().__init__()
        self.model = model
        self.use_obstacles = use_obstacles

    def forward(self, x_t, t, target, action, history, obstacle_centers=None):
        obstacles = None
        if self.use_obstacles:
            obstacles = [[{'center': center, 'radius': 0.5} for center in sample]
                         for sample in obstacle_centers]
        return self.model(x_t, t, target, action, history, obstacles)


def visualize(output_dir=None, depth=3, obstacles=False):
    torch.manual_seed(42)
    config = Config()
    config.enable_obstacle_encoding = obstacles
    model = TraceInputs(AeroDM(config).cpu().eval(), obstacles).eval()
    inputs = (
        torch.randn(1, config.seq_len, config.state_dim),
        torch.tensor([config.diffusion_steps // 2], dtype=torch.long),
        torch.tensor([[1., 2., 3., 1.]]),
        torch.nn.functional.one_hot(torch.tensor([0]), config.action_dim).float(),
        torch.randn(1, config.history_len, config.state_dim),
        torch.randn(1, 3, 3),
    )
    if not obstacles:
        inputs = inputs[:-1]
    output_dir = Path(output_dir) if output_dir else ROOT / 'output/network'
    output_dir.mkdir(parents=True, exist_ok=True)
    name = 'aerodm_obstacles' if obstacles else 'aerodm'
    # Disable fused attention to expose operations to torchview's tensor tracing.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            prediction = model(*inputs)
            assert prediction.shape == (1, config.seq_len, config.state_dim)
            graph = draw_graph(model, input_data=inputs, depth=depth,
                               expand_nested=True, roll=True, graph_name=name,
                               device='cpu', mode='eval')
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    graph.visual_graph.graph_attr.update(rankdir='TB')
    graph.visual_graph.node_attr.update(fontname='Helvetica')
    graph.visual_graph.save(str(output_dir / f'{name}.dot'))
    for fmt in ('svg', 'pdf'):
        path = graph.visual_graph.render(filename=name, directory=str(output_dir),
                                        format=fmt, cleanup=True)
        print(path)
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--obstacles', action='store_true')
    args = parser.parse_args()
    visualize(args.output_dir, args.depth, args.obstacles)
