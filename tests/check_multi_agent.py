"""Run: PYTHONPATH=. python tests/check_multi_agent.py"""
import torch
from demo_multi_agent import MultiAgentGuidance, pairwise_distances, sample_multi_agent
from AeroDM_ObstacleAware_Training_Test import AeroDM, Config


def main():
    mean = torch.zeros(1, 1, 11)
    std = torch.ones_like(mean)
    std[..., 1:4] = torch.tensor([2., 3., 4.])
    x = torch.zeros(3, 8, 10)  # All agents coincide: tie-breaking must separate them.
    guide = MultiAgentGuidance(mean, std, iterations=100)
    result = guide(x)
    assert torch.isfinite(result).all()
    assert pairwise_distances(result[..., 1:4] * std[..., 1:4]).min() > 0.99
    torch.testing.assert_close(result[..., [0, 4, 5, 6, 7, 8, 9]], x[..., [0, 4, 5, 6, 7, 8, 9]])
    torch.testing.assert_close(result[..., 1:4].mean(0), torch.zeros(8, 3), atol=1e-6, rtol=0)
    # Spatial intersections at DIFFERENT times should not cause repulsion.
    crossing = torch.zeros(2, 2, 10)
    crossing[0, :, 1] = torch.tensor([0., 2.])
    crossing[1, :, 1] = torch.tensor([2., 0.])
    torch.testing.assert_close(guide(crossing), crossing)
    torch.testing.assert_close(guide(x[:1]), x[:1])
    config = Config()
    config.latent_dim = 16
    config.num_layers = 1
    config.seq_len = 8
    config.history_len = 4
    config.diffusion_steps = 4
    model = AeroDM(config).eval()
    history = torch.randn(3, 4, 10)
    target = torch.zeros(3, 4)
    action = torch.zeros(3, config.action_dim)
    guide.calls = 0
    sample_multi_agent(model, history, target, action, guide, last_m=2)
    assert guide.calls == 2
    torch.manual_seed(7)
    a = sample_multi_agent(model, history, target, action, guide, last_m=0)
    torch.manual_seed(7)
    b = sample_multi_agent(model, history, target, action, None, last_m=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert guide.calls == 2
    print('PASS: separation, coincidence, physical scaling, same-time matching, M-step gating, M=0 equivalence')


if __name__ == '__main__':
    main()
