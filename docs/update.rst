.. include:: names.rst

.. toctree::
   :maxdepth: 2

Updating the App
================

Instructions vary depending on how you installed the application. Please locate the appropriate instructions below that fit your specific needs.

Installed using conda
---------------------

If you want to make absolutely sure that everything works as expected, you should always re-run the installation instructions to create an entirely new environment. This may also bear the benefit that your old version is still available if needed since, sometimes, as was the case from v1.0.x to v1.1.x, changes may be *breaking* changes. Meaning that data you created previously may not transfer 1:1 to the new version::

    # Windows, macOS
    conda create -n track -c trexing trex

    # Linux
    conda create -n track -c defaults -c conda-forge -c trexing trex

.. NOTE::
    In the case mentioned the change was from Tensorflow 1 to Tensorflow 2 with a new network architecture. This changes the way that visual identification weights are saved. You were not able to load old-format weights (from v1.0.x) anymore in the new version (>1.1).

conda-integrated update mechanism
---------------------------------

The ``conda update`` command might not work for |trex|, especially for major version changes, but if you want to try it anyway, open up an anaconda shell and activate your environment, e.g.::

    conda activate tracking

and now all you need to do is::

    # Windows, macOS
    conda update -c trexing trex

    # Linux
    conda update -c defaults -c conda-forge -c trexing trex

and follow the instructions on screen.

.. WARNING::
    Older versions of |trex| may use different python versions, which can cause the ``conda update`` process to fail since it does not automatically update python along with |trex|. In such a case, the easiest way is to reinstall the application as described in :doc:`install` and thus replace your current conda environment with a new one, or to manually mention the required python version as part of the update command, e.g.::

        conda update -c trexing trex python=3.9


Installed manually
------------------

If you compiled the software yourself, then you simply need to execute::

    git pull
    cd Application/build
    cmake --build .

Make sure that - if you customized the source code in your version - your changes do not block git from updating the repository in step one.
