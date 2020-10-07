from distutils import sysconfig
import os
import sys
import itertools

def get_python_library(python_version):
    """Get path to the python library associated with the current python
    interpreter."""
    # determine direct path to libpython
    python_library = sysconfig.get_config_var('LIBRARY')

    # if static (or nonexistent), try to find a suitable dynamic libpython
    if (python_library is None or
            os.path.splitext(python_library)[1][-2:] == '.a'):

        candidate_lib_prefixes = ['', 'lib']

        candidate_extensions = ['.lib', '.so', '.a', '.dll']
        if sysconfig.get_config_var('WITH_DYLD'):
            candidate_extensions.insert(0, '.dylib')

        candidate_versions = [python_version]
        if python_version:
            candidate_versions.append('')
            candidate_versions.insert(
                0, "".join(python_version.split(".")[:2]))

        abiflags = getattr(sys, 'abiflags', '')
        candidate_abiflags = [abiflags]
        if abiflags:
            candidate_abiflags.append('')

        # Ensure the value injected by virtualenv is
        # returned on windows.
        # Because calling `sysconfig.get_config_var('multiarchsubdir')`
        # returns an empty string on Linux, `du_sysconfig` is only used to
        # get the value of `LIBDIR`.
        candidate_arches = [""]
        libdir = sysconfig.get_config_var('LIBDIR')
        if sysconfig.get_config_var('MULTIARCH'):
            masd = sysconfig.get_config_var('multiarchsubdir')
            if masd:
                if masd.startswith(os.sep):
                    masd = masd[len(os.sep):]
                #libdir = os.path.join(libdir, masd)
                candidate_arches.append(masd)
            else:
                #libdir = os.path.join(libdir, sysconfig.get_config_var('MULTIARCH'))
                candidate_arches.append(sysconfig.get_config_var('MULTIARCH'))

        if libdir is None:
            libdir = os.path.abspath(os.path.join(
                sysconfig.get_config_var('LIBDEST'), "..", "libs"))
        candidates = (
            os.path.join(libdir, lib,
                ''.join((pre, 'python', ver, abi, ext))
            )
            for (lib, pre, ext, ver, abi) in itertools.product(
                candidate_arches,
                candidate_lib_prefixes,
                candidate_extensions,
                candidate_versions,
                candidate_abiflags
            )
        )

        for candidate in candidates:
            if os.path.exists(candidate):
                # we found a (likely alternate) libpython
                python_library = candidate
                break

    # TODO(opadron): what happens if we don't find a libpython?

    return python_library

print(get_python_library(".".join(sys.version.split('.')[:2])))
