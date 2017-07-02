#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/module.h>
#include <linux/types.h>
#include <asm/page.h>

MODULE_DESCRIPTION("VMA Test part1");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

int my_open(struct inode* inode, struct file* fp);
int my_release(struct inode* inode, struct file* fp);
int my_remap_mmap(struct file* fp, struct vm_area_struct* vma);
int my_nopage_mmap(struct file* fp, struct vm_area_struct* vma);

struct file_operations my_remap_ops = {
  open: my_open,
  release: my_release,
  mmap: my_remap_mmap,
};

struct file_operations my_nopage_ops = {
  open: my_open,
  release: my_release,
  mmap: my_nopage_mmap,
};

#define MAX_DEV 2

struct file_operations *my_fops[MAX_DEV] = {
  &my_remap_ops,
  &my_nopage_ops,
};

int my_open(struct inode* inode, struct file* fp) {
  unsigned int dev = MINOR(inode->i_rdev);

  if (dev >= MAX_DEV)
    return -ENODEV;

  fp->f_op = my_fops[dev];

//  Removed since 2.6.10...
//  MOD_INC_USE_COUNT;

  return 0;
}

int my_release(struct inode* inode, struct file* fp) {
  return 0;
}

void my_vma_open(struct vm_area_struct* vma) {
}

void my_vma_close(struct vm_area_struct* vma) {
}

static struct vm_operations_struct my_remap_vm_ops = {
  open: my_vma_open,
  close: my_vma_close,
};

static int __init do_setup(void) {
  return 0;
}

static void __exit do_cleanup(void) {
}

module_init(do_setup);
module_exit(do_cleanup);
