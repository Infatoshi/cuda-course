#include <stdio.h> // Standard input/output header file (for printf)

// & "address of" operator
// * "dereference" operator

int main() {
    int x = 10;
    int* ptr = &x; // & is used to get the memory address of a variable (x)
    printf("Address of x: %p\n", ptr);  // Output: memory address of x
    printf("Value of x: %d\n", *ptr);  // Output: 10
    // * in the prev line is used to get the value of 
    // the memory address stored in ptr (dereferencing)

}