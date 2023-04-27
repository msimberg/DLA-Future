# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# dlaf-no-license-check

from spack import *


class DlaFuture(CMakePackage, CudaPackage, ROCmPackage):
    """DLA-Future library: Distributed Linear Algebra with Future"""

    homepage = "https://github.com/eth-cscs/DLA-Future/wiki"
    git = "https://github.com/eth-cscs/DLA-Future"

    maintainers = ["teonnik", "albestro", "Sely85"]

    version("develop", branch="master")

    cxxstds = ('17', '20')
    variant('cxxstd',
            default='17',
            values=cxxstds,
            description='Use the specified C++ standard when building')
    conflicts('cxxstd=20', when='+cuda')

    variant("shared", default=True, description="Build shared libraries.")

    variant("doc", default=False, description="Build documentation.")

    variant("miniapps", default=False, description="Build miniapps.")

    variant("ci-test", default=False, description="Build for CI (Advanced usage).")
    conflicts('~miniapps', when='+ci-test')

    depends_on("cmake@3.22:", type="build")
    depends_on("doxygen", type="build", when="+doc")
    depends_on("mpi")
    depends_on("blaspp@2022.05.00:")
    depends_on("lapackpp@2022.05.00:")

    depends_on("umpire~examples")
    depends_on("umpire+cuda~shared", when="+cuda")
    depends_on("umpire+rocm~shared", when="+rocm")
    depends_on("umpire@4.1.0:")

    depends_on("pika@main")
    depends_on("pika-algorithms@0.1:")
    depends_on("pika +mpi")
    depends_on("pika +cuda", when="+cuda")
    depends_on("pika +rocm", when="+rocm")
    for cxxstd in cxxstds:
        depends_on("pika cxxstd={0}".format(cxxstd), when="cxxstd={0}".format(cxxstd))
        depends_on(
            "pika-algorithms cxxstd={0}".format(cxxstd),
            when="cxxstd={0}".format(cxxstd),
        )

    for build_type in ("Debug", "RelWithDebInfo", "Release"):
        depends_on(
            "pika build_type={0}".format(build_type),
            when="build_type={0}".format(build_type),
        )
        depends_on(
            "pika-algorithms build_type={0}".format(build_type),
            when="build_type={0}".format(build_type),
        )

    depends_on("whip +cuda", when="+cuda")
    depends_on("whip +rocm", when="+rocm")

    depends_on("rocblas", when="+rocm")
    depends_on("rocprim", when="+rocm")
    depends_on("rocsolver", when="+rocm")
    depends_on("rocthrust", when="+rocm")

    conflicts("+cuda", when="+rocm")

    with when("+rocm"):
        for val in ROCmPackage.amdgpu_targets:
            depends_on("pika amdgpu_target={0}".format(val),
                when="amdgpu_target={0}".format(val))
            depends_on("rocsolver amdgpu_target={0}".format(val),
                when="amdgpu_target={0}".format(val))
            depends_on("rocblas amdgpu_target={0}".format(val),
                when="amdgpu_target={0}".format(val))
            depends_on("rocprim amdgpu_target={0}".format(val),
                when="amdgpu_target={0}".format(val))
            depends_on("rocthrust amdgpu_target={0}".format(val),
                when="amdgpu_target={0}".format(val))
            depends_on("whip amdgpu_target={0}".format(val),
                when="amdgpu_target={0}".format(val))
            depends_on("umpire amdgpu_target={0}".format(val),
                when="amdgpu_target={0}".format(val))

    with when("+cuda"):
        for val in CudaPackage.cuda_arch_values:
            depends_on("pika cuda_arch={0}".format(val),
                when="cuda_arch={0}".format(val))
            depends_on("umpire cuda_arch={0}".format(val),
                when="cuda_arch={0}".format(val))

    def cmake_args(self):
        spec = self.spec
        args = []

        args.append(self.define_from_variant("BUILD_SHARED_LIBS", "shared"))

        # BLAS/LAPACK
        if "^mkl" in spec:
            args.append(self.define("DLAF_WITH_MKL", True))
        else:
            args.append(self.define("DLAF_WITH_MKL", False))
            args.append(self.define(
                    "LAPACK_LIBRARY",
                    " ".join([spec[dep].libs.ld_flags for dep in ["blas", "lapack"]]),
                ))

        # CUDA/HIP
        args.append(self.define_from_variant("DLAF_WITH_CUDA", "cuda"))
        args.append(self.define_from_variant("DLAF_WITH_HIP", "rocm"))
        if "+rocm" in spec:
            archs = self.spec.variants["amdgpu_target"].value
            if "none" not in archs:
                arch_str = ";".join(archs)
                args.append(self.define("CMAKE_HIP_ARCHITECTURES", arch_str))
        if "+cuda" in spec:
            archs = self.spec.variants["cuda_arch"].value
            if "none" not in archs:
                arch_str = ";".join(archs)
                args.append(self.define("CMAKE_CUDA_ARCHITECTURES", arch_str))

        # DOC
        args.append(self.define_from_variant("DLAF_BUILD_DOC", "doc"))

        if "+ci-test" in self.spec:
            # Enable TESTS and setup CI specific parameters
            args.append(self.define("CMAKE_CXX_FLAGS", "-Werror"))
            if "+cuda":
                args.append(self.define("CMAKE_CUDA_FLAGS", "-Werror=all-warnings"))
            if "+rocm":
                args.append(self.define("CMAKE_HIP_FLAGS", "-Werror"))
            args.append(self.define("BUILD_TESTING", True))
            args.append(self.define("DLAF_BUILD_TESTING", True))
            args.append(self.define("DLAF_CI_RUNNER_USES_MPIRUN", True))
        else:
            # TEST
            args.append(self.define("DLAF_BUILD_TESTING", self.run_tests))

        # MINIAPPS
        args.append(self.define_from_variant("DLAF_BUILD_MINIAPPS", "miniapps"))

        return args
