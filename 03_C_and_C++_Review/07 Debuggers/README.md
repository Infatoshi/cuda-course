## GDB Debugger for C/C++ (not covered in video lectures)

> Install -> `sudo apt install gdb`

Watch: https://www.youtube.com/watch?v=Dq8l1_-QgAc

### Commands
- run or r: Executes the program from start to end.
- break or b: Set a breakpoint on a particular line.
- disable: Disables a breakpoint.
- enable: Enables a disabled breakpoint.

- next or n: Executes the next line of code in C.
- nexti: Executes the next instruction in assembly.
- step or s: Executes the next line of code, but if the next line of code is a function, it will enter the function and stop at the first line of the function.
- stepi: Executes the next instruction in assembly, but if the next instruction is a function, it will enter the function and stop at the first instruction of the function.

- list or l: Shows all the code within current scope.
- print or p: Prints the value of a variable.
- quit or q: Exits gdb.
- clear: Removes all breakpoints.
- continue or c: Continues the execution of the program until the next breakpoint.

> Its worth noting that gdb is used for debugging C/C++ programs, whereas cuda-gdb is used for debugging CUDA programs.
