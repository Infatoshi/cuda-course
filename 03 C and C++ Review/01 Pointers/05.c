#include <stdio.h>

int main() {
    int arr[] = {12, 24, 36, 48, 60};

    int* ptr = arr;  // ptr points to the first element of arr (default in C)

    printf("Position one: %d\n", *ptr);  // Output: 12

    for (int i = 0; i < 5; i++) {
        printf("%d\t", *ptr);
        printf("%p\t", ptr);
        ptr++;
    }
    // Output: 
    // Position one: 12
    // disclaimer: the memory addresses won't be the same each time you run
    // 12 0x7fff773fe780
    // 24 0x7fff773fe784
    // 36 0x7fff773fe788
    // 48 0x7fff773fe78c
    // 60 0x7fff773fe790

    // notice that the pointer is incremented by 4 bytes (size of int = 4 bytes * 8 bits/bytes = 32 bits = int32) each time. 
    // ptrs are 64 bits in size (8 bytes). 2**32 = 4,294,967,296 which is too small given how much memory we typically have.
    // arrays are layed out in memory in a contiguous manner (one after the other rather than at random locations in the memory grid)

}