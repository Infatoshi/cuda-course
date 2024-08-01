# Extras

## How does CUDA handle conditional if/else logic?
- CUDA does not handle conditional if/else logic well. If you have a conditional statement in your kernel, the compiler will generate code for both branches and then use a predicated instruction to select the correct result. This can lead to a lot of wasted computation if the branches are long or if the condition is rarely met. It is generally a good idea to try to avoid conditional logic in your kernels if possible.
- If it is unavoidable, you can dig down to the PTX assembly code (`nvcc -ptx kernel.cu -o kernel`) and see how the compiler is handling it. Then you can look into the compute metrics of the instructions used and try to optimize it from there.
- Single thread going down a long nested if else statement will look more serialized and leave the other threads waiting for the next instruction while the single threads finishes. this is called **warp divergence** and is a common issue in CUDA programming when dealing with threads specifically within a warp.
- vector addition is fast because divergence isn’t possible, not a different possible way for instructions to carry out.