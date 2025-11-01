# Custom PyTorch Extensions

```bash
python setup.py install 
```

## What is the `scalar_t` type?
- think of this as the type of the elements in the CUDA torch tensor
- it gets safely compiled down to the appropriate type for the GPU (fp32 or fp64)

## Why use `__restrict__`?
```cpp
// because this code behaves a certain way

void add_arrays(int* a, int* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] + b[i];
    }
}

int main() {
    int data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Overlapping call
    add_arrays(data, data + 3, 7);
    
    // Print result
    for (int i = 0; i < 10; i++) {
        printf("%d ", data[i]);
    }
    return 0;
}
```

```python
# Initial state of the 'data' array:
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Memory layout visualization:
#  a (data)     b (data + 3)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#  ^        ^
#  |        |
#  a[0]     b[0]

# After i = 0: data[0] = data[0] + data[3]
[5, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ^

# After i = 1: data[1] = data[1] + data[4]
[5, 7, 3, 4, 5, 6, 7, 8, 9, 10]
#    ^

# After i = 2: data[2] = data[2] + data[5]
[5, 7, 9, 4, 5, 6, 7, 8, 9, 10]
#       ^

# After i = 3: data[3] = data[3] + data[6]
[5, 7, 9, 11, 5, 6, 7, 8, 9, 10]
#          ^

# After i = 4: data[4] = data[4] + data[7]
# Note: data[4] is now changed from its original value!
[5, 7, 9, 11, 13, 6, 7, 8, 9, 10]
#              ^

# After i = 5: data[5] = data[5] + data[8]
[5, 7, 9, 11, 13, 15, 7, 8, 9, 10]
#                  ^

# After i = 6: data[6] = data[6] + data[9]
[5, 7, 9, 11, 13, 15, 17, 8, 9, 10]
#                      ^

# Final state of the 'data' array:
data = [5, 7, 9, 11, 13, 15, 17, 8, 9, 10]
```

## Torch Binding section
```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polynomial_activation", &polynomial_activation_cuda, "Polynomial activation (CUDA)");
}
```

This section uses pybind11 to create a Python module for the CUDA extension:
- PYBIND11_MODULE is a macro that defines the entry point for the Python module.
- TORCH_EXTENSION_NAME is a macro defined by PyTorch that expands to the name of the extension (usually derived from the setup.py file).
- m is the module object being created.
- m.def() adds a new function to the module:
  - The first argument "polynomial_activation" is the name of the function in Python.
  - &polynomial_activation_cuda is a pointer to the C++ function to be called.
  - The last argument is a docstring for the function.

> we essentially tell the compiler that the arrays are not overlapping
> this way the compiler can make assumptions about the memory layout and 
> aggressively optimize
- notice in the top line how this is saved to `/home/elliot/.cache/torch_extensions/py311_cu121` (you can remove stuff in the .cache directory if it gets flooded with binaries)


## Learning Resources
- https://github.com/pytorch/extension-cpp
- https://pytorch.org/tutorials/advanced/cpp_custom_ops.html
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://pytorch.org/docs/stable/notes/extending.html
