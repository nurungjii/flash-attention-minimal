#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

#include <torch/extension.h>

// torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("mha_forward", torch::wrap_pybind_function(forward), "mha_forward");
// }

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

// namespace minimal_attn {

// // Following flash.cu
// TORCH_LIBRARY_IMPL(minimal_attn, CUDA, m) {
//   m.impl("mha_forward", &forward);
// }


// }