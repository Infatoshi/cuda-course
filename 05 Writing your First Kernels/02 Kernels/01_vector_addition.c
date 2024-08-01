#include <stdio.h>

int main() {
    int n = 10;
    int a[n], b[n], c[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    return 0;

}