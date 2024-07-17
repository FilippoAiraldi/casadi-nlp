Installation
============

Using `pip`
------------

You can use `pip` to install **csnlp** with the command

.. code:: bash

   pip install csnlp

**csnlp** has the following dependencies

-  Python 3.9 or higher
-  `NumPy <https://pypi.org/project/numpy/>`__
-  `CasADi <https://pypi.org/project/casadi/>`__
-  `Joblib <https://joblib.readthedocs.io/>`__


Using source code
-----------------

If you'd like to play around with the source code instead, run

.. code:: bash

   git clone https://github.com/FilippoAiraldi/casadi-nlp.git

The `main` branch contains the main releases of the packages (and the occasional post
release). The `experimental` branch is reserved for the implementation and test of new
features and hosts the release candidates. You can then install the package to edit it
as you wish as

.. code:: bash

   pip install -e /path/to/casadi-nlp
