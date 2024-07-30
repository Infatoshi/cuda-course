### Makefiles
```make
targets: prerequisites
    bash command
    possibly another bash command?
```

### Purpose of CMakeLists.txt?

CMake is a tool that **generates Makefiles**. It is a build system generator. It is used to build, test, and package software. It is a cross-platform build system. It is used to control the software compilation process using simple platform and compiler independent configuration files.



### What does `.PHONY` do?

Say we have a Makefile with a target named 'clean':
```make
clean:
    rm -rf build/*
```

Suppose we have a directory named 'clean' in the same directory as the Makefile. If we run `make clean`, make will not run the command in the target 'clean'. Instead, it will see that the directory 'clean' already exists and will not run the command.

In short, we essentially make a bunch of mappings from target names to commands. If we have a file or directory with the same name as a target, make will not run the command. This is where `.PHONY` comes in.

### `:=` vs `=` in Makefiles

`=` is used for defining variables. It is called a **recursive assignment**. The value of the variable is re-evaluated each time the variable is used.

`:=` is used for defining variables. It is called a **simple assignment** or **immediate assignment**. The value of the variable is evaluated only once, at the point of definition.

Example:
```make
A = $(B)
B = hello
C := $(B)
B = world

all:
    @echo A is $(A)  # Outputs: A is world
    @echo C is $(C)  # Outputs: C is hello
```

### What is the purpose of `@` in Makefiles?

The @ symbol prevents the command itself from being echoed to the console when the Makefile is executed.

Example:
```make
clean:
    rm -rf build/*
```

```bash
$ make clean
rm -rf build/*
```

```make
clean:
    @rm -rf build/*
```

```bash
$ make clean
$
```

