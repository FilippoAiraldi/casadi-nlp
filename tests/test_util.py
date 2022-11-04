import unittest
import casadi as cs
from casadi_mpc.util import is_casadi_object


class TestUtil(unittest.TestCase):
    def test_is_casadi_object__guesses_correctly(self):
        for o, flag in [
            (5.0, False),
            (unittest.TestCase(), False),
            (cs.DM(5), True),
            (cs.SX.sym('x'), True),
            (cs.MX.sym('x'), True),
        ]:
            self.assertEqual(is_casadi_object(o), flag)


if __name__ == '__main__':
    unittest.main()
