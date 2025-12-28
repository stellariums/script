import math
import unittest

import numpy as np
from ase import Atoms

from avgangle import average_bond_angle, compute_all_angles


class TestAverageBondAngle(unittest.TestCase):
    """针对平均键角计算的单元测试。"""

    def test_linear_three_atoms(self):
        """
        测试简单的线性三原子体系，理论键角应为 180 度。
        """
        # 三个原子排成一条直线：C(0,0,0) - C(1,0,0) - C(2,0,0)
        atoms = Atoms("CCC", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        cutoff = 1.5

        angles = compute_all_angles(atoms, cutoff=cutoff)
        self.assertGreaterEqual(angles.size, 1)
        mean_angle = float(np.mean(angles))
        self.assertAlmostEqual(mean_angle, 180.0, places=6)

        avg_angle = average_bond_angle(atoms, cutoff=cutoff)
        self.assertAlmostEqual(avg_angle, 180.0, places=6)

    def test_no_neighbor_returns_nan(self):
        """
        当结构中无法构成任何键角时，应返回 NaN。
        """
        atoms = Atoms("C", positions=[(0, 0, 0)])
        cutoff = 1.0

        angles = compute_all_angles(atoms, cutoff=cutoff)
        self.assertEqual(angles.size, 0)

        avg_angle = average_bond_angle(atoms, cutoff=cutoff)
        self.assertTrue(math.isnan(avg_angle))


if __name__ == "__main__":
    unittest.main()
