/* Minimal probe: which OpenMP for-loop forms does this MSVC accept?
 * C3015 was raised for `for (int ib = 0; ib < nblocks; ib++)`, which is the canonical
 * form, so something else about the surrounding code is responsible. Each variant is
 * compiled separately so one failure does not hide the others.
 */
#include <omp.h>
#include <stdio.h>

int main(void) {
    int n = 64;

    /* A: plain literal bound, variable declared in the for-init */
#pragma omp parallel for
    for (int i = 0; i < 64; i++) {
        n += 0;
    }

    /* B: bound from a const int local */
    const int nb = 16;
#pragma omp parallel for
    for (int i = 0; i < nb; i++) {
        n += 1;
    }

    /* C: bound from a non-const local, with schedule */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        n += 0;
    }

    /* D: loop variable declared before the pragma */
    int k;
#pragma omp parallel for
    for (k = 0; k < nb; k++) {
        n += 1;
    }

    printf("variants compiled, threads=%d, n=%d\n", omp_get_max_threads(), n);
    return 0;
}
