/**
 * 1. f is a function,
 * 2. wchich returns a pointer,
 * 3. which is pointing to int.
 */
int *f();
/**
 * 1. pf is a pointer,
 * 2. which is pointing to a function,
 * 3. whicch is returning int.
 */
int (*pf)();
/**
 * 1. argv is a pointer,
 * 2. which is pointing to pointer,
 * 3. which is pointing to char.
 */
char **argv;
/**
 * 1. t is an array,
 * 2. which elements are poiners,
 * 3. which are pointing to int.
 */
int* t[5];
/**
 * 1. pt is an pointer,
 * 2. which is pointing to an array,
 * 3. which elements are int.
 */
int (*pt)[5];
/**
 * 1. x is a function,
 * 2. which returns pointer,
 * 3. which is pointing to array,
 * 4. which elements are pointers,
 * 5. which are pointing to functions,
 * 6. which return char
 */
char (*(*x())[]);
/**
 * 1. y is an array,
 * 2. which elements are pointers,
 * 3. which are pointing to functions,
 * 4. which return pointers,
 * 5. which are pointing to arrays,
 * 6. which elements are char
 */
char (*(*y[5])())[5];

auto main() -> decltype(0) {
  return 0;
}
