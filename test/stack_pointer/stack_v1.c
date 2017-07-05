#include <linux/kernel.h>
#include <linux/module.h>

MODULE_DESCRIPTION("Stack Pointer Test");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

int sp3;

int f() {
  int sp2;
  return sp2;
}

static int __init do_init(void) {
  int sp1;
  int sp4 = f();
  printk(KERN_INFO "int sp1 @ %p\n", &sp1);
  printk(KERN_INFO "int sp3 @ %p\n", &sp3);
  printk(KERN_INFO "int sp4 @ %p\n", &sp4);
  return 0;
}

static void __exit do_cleanup(void) {} // Nothing

module_init(do_init);
module_exit(do_cleanup);
