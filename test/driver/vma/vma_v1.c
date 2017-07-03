#include <linux/debugfs.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/types.h>
#include <linux/version.h>
#include <asm/page.h>

MODULE_DESCRIPTION("VMA Test part1");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

struct dentry* mine;

#define ME "my_vma"
#define PRINT() printk(KERN_NOTICE ME ": %s\n", __func__);

struct my_info {
  char* data;
  int ref;
};

void mmap_open(struct vm_area_struct* vma) {
  struct my_info* me = (struct my_info*)vma->vm_private_data;

  PRINT();

  me->ref++;
}

void mmap_close(struct vm_area_struct* vma) {
  struct my_info* me = (struct my_info*)vma->vm_private_data;

  PRINT();

  me->ref--;
}

int mmap_fault(struct vm_area_struct* vma, struct vm_fault* vmf) {
  struct page* p;
  struct my_info* me;
  unsigned long addr = (unsigned long)vmf->virtual_address;

  PRINT();

  if (addr > vma->vm_end) {
    printk(KERN_WARNING "Invalid address");
    return VM_FAULT_SIGBUS;
  }

  me = (struct my_info*)vma->vm_private_data;
  if (!me->data) {
    printk(KERN_WARNING "No data");
    return VM_FAULT_SIGBUS;
  }

  p = virt_to_page(me->data);

  get_page(p);

  vmf->page = p;
  return 0;
}

struct vm_operations_struct mmap_vm_ops = {
  .open = mmap_open,
  .close = mmap_close,
  .fault = mmap_fault,
};

int my_mmap(struct file* fp, struct vm_area_struct* vma) {
  vma->vm_ops = &mmap_vm_ops;
//  vma->vm_flags |= VM_RESERVED;
  vma->vm_private_data = fp->private_data;

  PRINT();

  mmap_open(vma);
  return 0;
}

int my_close(struct inode* inode, struct file* fp) {
  struct my_info* me = fp->private_data;

  PRINT()

  free_page((unsigned long)me->data);
  kfree(me);
  fp->private_data = NULL;
  return 0;
}

int my_open(struct inode* inode, struct file* fp) {
  struct my_info* me;
  me = kmalloc(sizeof(struct my_info), GFP_KERNEL);
  // Get new memory
  me->data = (char*)get_zeroed_page(GFP_KERNEL);

  PRINT();

  memcpy(me->data, "ABCDEFG: ", 32);
  memcpy(me->data + 32, fp->f_dentry->d_name.name,
    strlen(fp->f_dentry->d_name.name));

  fp->private_data = me;
  return 0;
}

static const struct file_operations my_ops = {
  .open = my_open,
  .release = my_close,
  .mmap = my_mmap,
};

static int __init do_setup(void) {
  mine = debugfs_create_file(ME, 0644, NULL, NULL, &my_ops);
  PRINT();
  return 0;
}

static void __exit do_cleanup(void) {
  PRINT();
  debugfs_remove(mine);
}

module_init(do_setup);
module_exit(do_cleanup);
