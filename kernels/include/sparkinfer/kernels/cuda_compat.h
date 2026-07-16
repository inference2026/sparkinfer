#pragma once

// MSVC + NVCC on Windows rejects explicit `template __global__` instantiations
// ("no instance of function template matches the specified type") even when the
// launch site is in the same translation unit. Implicit instantiation from the
// <<<>>> launch is sufficient there. Linux/GCC keeps explicit inst for link hygiene.
#if defined(_MSC_VER)
#define SPARKINFER_KERNEL_INST(...) /* omitted on MSVC */
#else
#define SPARKINFER_KERNEL_INST(...) __VA_ARGS__
#endif
