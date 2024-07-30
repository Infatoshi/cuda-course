#include <stdio.h>

int main() {
    int num = 10;
    float fnum = 3.14;
    void* vptr;

    vptr = &num;
    printf("Integer: %d\n", *(int*)vptr);  // Output: 10

    vptr = &fnum;
    printf("Float: %.2f\n", *(float*)vptr);  // Output: 3.14
}
