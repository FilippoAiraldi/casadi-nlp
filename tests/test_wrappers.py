import unittest
from casadi_nlp import Nlp
from casadi_nlp.wrappers import Wrapper, DifferentiableNlp


class TestWrapper(unittest.TestCase):
    def test_unwrapped__unwraps_nlp_correctly(self):
        nlp = Nlp()
        self.assertIs(nlp, nlp.unwrapped)
        wrapped = Wrapper[Nlp](nlp)
        self.assertIs(nlp, wrapped.unwrapped)
        
        
if __name__ == '__main__':
    unittest.main()
