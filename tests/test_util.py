import unittest
from functools import cached_property
import casadi as cs
from casadi_mpc.util import is_casadi_object, cached_property_reset


class DummyWithCachedProperty:
    def __init__(self) -> None:
        self.counter = 0

    @cached_property
    def a_cached_property(self) -> int:
        self.counter += 1
        return self.counter

    @cached_property_reset(a_cached_property)
    def clear_cache(self) -> None:
        return


class TestUtil(unittest.TestCase):
    def test_cached_property_reset__raises__with_invalid_type(self):
        with self.assertRaises(TypeError):
            cached_property_reset(5)

    def test_cached_property_reset__clears_property_cache(self):
        dummy = DummyWithCachedProperty()
        dummy.a_cached_property
        dummy.a_cached_property
        self.assertEqual(dummy.counter, 1)
        dummy.clear_cache()
        dummy.a_cached_property
        self.assertEqual(dummy.counter, 2)
        dummy.clear_cache()
        dummy.a_cached_property
        dummy.a_cached_property
        self.assertEqual(dummy.counter, 3)

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
