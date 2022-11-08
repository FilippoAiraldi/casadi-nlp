from typing import Dict
import unittest
from functools import cached_property
import casadi as cs
from casadi_nlp.util import (
    is_casadi_object, cache_clearer,
    dict2struct, struct_symSX, DMStruct,
    np_random
)
import numpy as np


class DummyWithCachedProperty:
    def __init__(self) -> None:
        self.counter1 = 0
        self.counter2 = 0

    @cached_property
    def prop1(self) -> int:
        self.counter1 += 1
        return self.counter1

    @cached_property
    def prop2(self) -> int:
        self.counter2 += 1
        return self.counter2

    @cache_clearer(prop1, prop2)
    def clear_cache(self) -> None:
        return


class DummyWithCachedProperty2(DummyWithCachedProperty):
    def __init__(self) -> None:
        super().__init__()
        self.counter3 = 0

    @cached_property
    def prop3(self) -> int:
        self.counter3 += 1
        return self.counter3

    @cache_clearer(prop3)
    def clear_cache(self) -> None:
        return super().clear_cache()


class TestUtil(unittest.TestCase):
    def test_CacheClearer__raises__with_invalid_type(self):
        with self.assertRaises(TypeError):
            cache_clearer(5)

    def test_CacheClearer__clears_property_cache(self):
        dummy = DummyWithCachedProperty()
        dummy.prop1
        dummy.prop1
        dummy.prop2
        dummy.prop2
        self.assertEqual(dummy.counter1, 1)
        self.assertEqual(dummy.counter2, 1)
        dummy.clear_cache()
        dummy.prop1
        dummy.prop2
        self.assertEqual(dummy.counter1, 2)
        self.assertEqual(dummy.counter2, 2)
        dummy.clear_cache()
        dummy.prop1
        dummy.prop1
        dummy.prop2
        dummy.prop2
        self.assertEqual(dummy.counter1, 3)
        self.assertEqual(dummy.counter2, 3)

    def test_CacheClearer__accepts_new_caches_to_clear(self):
        dummy = DummyWithCachedProperty2()
        dummy.prop1
        dummy.prop1
        dummy.prop2
        dummy.prop2
        dummy.prop3
        dummy.prop3
        self.assertEqual(dummy.counter1, 1)
        self.assertEqual(dummy.counter2, 1)
        self.assertEqual(dummy.counter3, 1)
        dummy.clear_cache()
        dummy.prop1
        dummy.prop2
        dummy.prop3
        self.assertEqual(dummy.counter1, 2)
        self.assertEqual(dummy.counter2, 2)
        self.assertEqual(dummy.counter3, 2)

    def test_is_casadi_object__guesses_correctly(self):
        for o, flag in [
            (5.0, False),
            (unittest.TestCase(), False),
            (cs.DM(5), True),
            (cs.SX.sym('x'), True),
            (cs.MX.sym('x'), True),
        ]:
            self.assertEqual(is_casadi_object(o), flag)

    def test_dict2struct__with_DM__returns_numerical_struct(self):
        d = {
            'x': cs.DM(np.random.rand(3, 2)),
            'y': cs.DM(np.random.rand(4, 1)),
        }
        s = dict2struct(d)
        self.assertIsInstance(s, DMStruct)
        for name, x in d.items():
            np.testing.assert_allclose(x, s[name])

    def test_dict2struct__with_SX__returns_sym_struct(self):
        d = {
            'x': cs.SX.sym('x', 3, 2),
            'y': cs.SX.sym('y', 4, 1),
        }
        s = dict2struct(d)
        self.assertIsInstance(s, struct_symSX)
        for name, x in d.items():
            self.assertTrue(cs.is_equal(s[name], x))

    def test_dict2struct__with_MX__returns_copy_of_dict(self):
        d = {
            'x': cs.MX.sym('x'),
            'y': cs.MX.sym('y'),
        }
        s = dict2struct(d)
        self.assertIsInstance(s, Dict)
        self.assertDictEqual(d, s)

    def test_np_random__raises__with_invalid_seed(self):
        with self.assertRaises(ValueError):
            np_random(-1)

    def test_np_random__initializes_rng_with_correct_seed(self):
        for seed in (69, None):
            rng, actual_seed = np_random(seed)
            self.assertIsInstance(rng, np.random.Generator)
            if seed is not None:
                self.assertEqual(seed, actual_seed)
            else:
                self.assertIsInstance(actual_seed, int)


if __name__ == '__main__':
    unittest.main()
