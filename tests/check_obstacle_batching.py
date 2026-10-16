"""Numerical equivalence checks; run with PYTHONPATH=. python tests/check_obstacle_batching.py."""
import copy
import time
import torch
import AeroDM_ObstacleAware_Training_Test as m


def main():
    torch.manual_seed(17)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = m.Config()
    c.dropout = 0.0  # Compare exact computations without differing dropout draws.
    legacy = m.AttentionObstacleEncoder(c).to(device)
    batched = copy.deepcopy(legacy)
    centers = torch.randn(4, 5, 3, device=device)
    radii = torch.rand(4, 5, device=device) + 0.5
    valid = torch.arange(5, device=device)[None] < torch.tensor([0, 1, 3, 5], device=device)[:, None]
    packed = dict(centers=centers, radii=radii, valid_mask=valid)
    lists = [[dict(center=centers[b, j], radius=float(radii[b, j]))
              for j in range(5) if valid[b, j]] for b in range(4)]
    expected = legacy(lists)
    actual = batched(packed)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    weight = torch.randn_like(actual)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    for p, q in zip(legacy.parameters(), batched.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4)
    criterion = m.AeroDMLoss(c, enable_obstacle_term=True)
    x = torch.randn(4, 60, 10, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mean = torch.randn(1, 1, 10, device=device)
    std = torch.rand_like(mean) + 0.5
    loss_a = criterion.compute_obstacle_distance_loss(x, lists, mean, std)
    loss_b = criterion.compute_obstacle_distance_loss(y, packed, mean, std)
    torch.testing.assert_close(loss_a, loss_b)
    loss_a.backward(); loss_b.backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)
    empty = dict(centers=centers, radii=radii, valid_mask=torch.zeros_like(valid))
    output = batched(empty)
    assert torch.isfinite(output).all() and (output == 0).all()
    output.sum().backward()
    assert criterion.compute_obstacle_distance_loss(x, empty, mean, std) == 0
    pos = torch.randn(256, 60, 3, device=device)
    generated = m.generate_training_obstacles_batch(pos)
    assert ((generated['valid_mask'].sum(1) >= 3) & (generated['valid_mask'].sum(1) <= 5)).all()
    assert ((generated['radii'] >= 0.5) & (generated['radii'] <= 1)).all()
    nearest = torch.cdist(generated['centers'] - 0.3, pos[:, 10:]).amin(-1)
    assert (nearest < 0.003).all()
    # End-to-end training step with the real configuration and obstacle batch.
    c = m.Config()
    model = m.AeroDM(c).to(device)
    mean = torch.zeros(1, 1, 11, device=device); std = torch.ones_like(mean)
    model.set_normalization_params(mean, std)
    x = torch.randn(c.batch_size, c.seq_len, c.state_dim, device=device)
    history = torch.randn(c.batch_size, c.history_len, c.state_dim, device=device)
    target = m.generate_target_waypoints(x)
    action = torch.nn.functional.one_hot(torch.zeros(c.batch_size, dtype=torch.long, device=device), c.action_dim).float()
    t = torch.randint(c.diffusion_steps, (c.batch_size,), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        obs = m.generate_training_obstacles_batch(x[..., 1:4])
        pred = model(x, t, target, action, history, obstacles_data=obs)
        loss = criterion(pred, x, obs, mean, std, history)[0]
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print('PASS: encoder outputs/parameter gradients, obstacle loss/gradients, empty masks, sampling rules, and full training steps')


if __name__ == '__main__':
    main()
