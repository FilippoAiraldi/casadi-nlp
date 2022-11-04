from typing import Tuple
import unittest
from casadi_mpc.debug import MpcDebug, MpcDebugEntry
import random


class TestMpcDebug(unittest.TestCase):
    def test_register__adds_correct_info(self):
        debug = MpcDebug()
        group = random.choice(['x', 'g', 'h'])
        name = 'var'
        shape = (2, 3)
        debug.register(group, name, shape)
        info: Tuple[range, MpcDebugEntry] = getattr(debug, f'_{group}_info')[0]
        self.assertEqual(info[0], range(shape[0] * shape[1]))
        self.assertEqual(info[1].name, name)
        self.assertEqual(info[1].shape, shape)
        self.assertEqual(info[1].type, MpcDebug.types[group])

    def test_register__raises__with_invalid_group(self):
        debug = MpcDebug()
        while True:
            group = chr(random.randint(ord('a'), ord('z')))
            if group not in debug.types.keys():
                break
        with self.assertRaises(AttributeError):
            debug.register(group, 'var1', (1, 2))

    def test_xhg_describe__gets_corret_variables(self):
        debug = MpcDebug()
        for group in debug.types.keys():
            debug.register(group, 'var1', (3, 3))
            debug.register(group, 'var2', (1, 1))
            info1: MpcDebugEntry = getattr(debug, f'{group}_describe')(0)
            info2: MpcDebugEntry = getattr(debug, f'{group}_describe')(9)
            self.assertEqual(info1.name, 'var1')
            self.assertEqual(info2.name, 'var2')

    def test_xhg_describe__raises__with_outofbound_index(self):
        debug = MpcDebug()
        for group in debug.types.keys():
            debug.register(group, 'var1', (1, 1))
            debug.register(group, 'var2', (1, 1))
            with self.assertRaises(IndexError):
                getattr(debug, f'{group}_describe')(10_000)


if __name__ == '__main__':
    unittest.main()
