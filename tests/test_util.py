from typing import Dict
import unittest
from functools import cached_property
import casadi as cs
from casadi_mpc.util import \
    is_casadi_object, cached_property_reset, dict2struct, struct_symSX


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

    def test_dict2struct__with_MX__returns_copy_of_dict(self):
        d = {
            'x': cs.SX.sym('x'),
            'y': cs.MX.sym('y'),
        }
        s = dict2struct(d)
        self.assertIsInstance(s, Dict)
        self.assertDictEqual(d, s)
        
    def test_dict2struct__with_SX__returns_struct(self):
        d = {
            'x': cs.SX.sym('x', 3, 2),
            'y': cs.SX.sym('y', 4, 1),
        }
        s = dict2struct(d)
        self.assertIsInstance(s, struct_symSX)
        for name, x in d.items():
            self.assertTrue(cs.is_equal(s[name], x))

if __name__ == '__main__':
    unittest.main()
