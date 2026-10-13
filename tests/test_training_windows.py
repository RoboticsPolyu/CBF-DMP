"""Test sampling without importing the training script's plotting dependencies."""
import ast
from pathlib import Path
import unittest
import torch

source_path = Path(__file__).resolve().parents[1] / 'AeroDM_ObstacleAware_Training_Test.py'
tree = ast.parse(source_path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_trajectory_windows')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
sample = namespace['sample_trajectory_windows']


class TrainingWindowTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.stack((torch.arange(160), torch.arange(160) + 1000)).float().unsqueeze(-1)

    def test_uniform_expansion_and_order(self):
        windows = sample(self.x, 20, 60, stride=5)
        self.assertEqual(windows.shape, (34, 80, 1))
        for source in range(2):
            for index, start in enumerate(range(0, 81, 5)):
                self.assertTrue(torch.equal(windows[source * 17 + index], self.x[source, start:start+80]))

    def test_off_grid_tail_is_not_added(self):
        windows = sample(self.x, 20, 60, stride=7)
        self.assertEqual(windows.shape, (24, 80, 1))
        self.assertEqual(windows[11, 0, 0].item(), 77)

    def test_random_evaluation_keeps_one_window_per_source(self):
        torch.manual_seed(42)
        windows = sample(self.x, 20, 60)
        self.assertEqual(windows.shape, (2, 80, 1))
        for index in range(2):
            start = int(windows[index, 0, 0]) - 1000 * index
            self.assertTrue(0 <= start <= 80)
            self.assertTrue(torch.equal(windows[index], self.x[index, start:start+80]))

    def test_exact_length_and_invalid_inputs(self):
        for stride in [None, 5, 200]:
            self.assertTrue(torch.equal(sample(self.x[:, :80], 20, 60, stride), self.x[:, :80]))
        for stride in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                sample(self.x, 20, 60, stride)
        with self.assertRaises(ValueError):
            sample(self.x[:, :79], 20, 60, 5)


if __name__ == '__main__':
    unittest.main()
