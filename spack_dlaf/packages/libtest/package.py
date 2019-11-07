# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install libtest
#
# You can edit this file again by typing:
#
#     spack edit libtest
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *

class Libtest(CMakePackage):
    """LIBTEST: an ICL library for SLATE project."""

    homepage = "https://bitbucket.org/icl/libtest"
    hg       = "https://bitbucket.org/icl/libtest"

    maintainers = ['Sely85']

    version('develop')