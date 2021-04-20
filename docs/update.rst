.. include:: names.rst

.. toctree::
   :maxdepth: 2

Updating the App
================

Instructions vary depending on how you installed the application. Please locate the appropriate instructions below that fit your specific needs.

Installed using conda
---------------------

Simply open up an anaconda shell and activate your environment, e.g.::

    conda activate tracking

and now all you need to do is::

    conda update -c trexing trex

and follow the instructions on screen.

.. WARNING::
    Older versions of |trex| may use different python versions, which can cause the ``conda update`` process to fail since it does not automatically update python along with |trex|. In such a case, the easiest way is to reinstall the application as described in :doc:`install` and thus replace your current conda environment with a new one, or to manually mention the required python version as part of the update command, e.g.::

        conda update -c trexing trex python=3.7