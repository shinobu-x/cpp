#include <linux/module.h>
#include <linux/mm.h>
#include <asm/pgtable.h>
#include <linux/fs.h>

#include "common.h"

void x_vma_open(struct vm_area_struct* vma) {
  struct x_dev_t* dev = vma->vm_private_data;
  dev->vmas++;
}

void x_vma_close(struct vm_area_struct* vma) {
  struct x_dev_t* dev = vma->vm_private_data;
  dev->vmas--;
}

static int x_vma_nopage(struct vm_area_struct* vma, struct vm_fault* vmf) {
  unsigned long offset;
  struct x_dev_t* ptr;
  struct x_dev_t* dev = vma->vm_private_data;
  struct page* page = NULL;
  void* pageptr = NULL;
  int r = VM_FAULT_NOPAGE;

  down(&dev->sem);
  offset = (unsigned long)(vmf->virtual_address - vma->vm_start) +
    (vma->vm_pgoff << PAGE_SHIFT);

  if (offset >= dev->size)
    goto out;  // Out of range

  offset >>= PAGE_SHIFT;  // Offset is a number of pages

  for (ptr = dev; ptr && offset >= dev->qset;) {
    ptr = ptr->next;
    offset -= dev->qset;
  }

  if (ptr && ptr->data)
    pageptr = ptr->data[offset];

  if (!pageptr)
    goto out;  // Hole or end of file

  page = virt_to_page(pageptr);

  get_page(page);
  vmf->page = page;
  r = 0;

out:
  up(&dev->sem);
  return r;
}

struct vm_operations_struct ops_vm_t = {
  .open = x_vma_open,
  .close = x_vma_close,
  .fault = x_vma_nopage,
};

int x_mmap(struct file* fp, struct vm_area_struct* vma) {
  struct inode* inode = fp->f_path.dentry->d_inode;

  if (devs_t[iminor(inode)].order)
    return -ENODEV;

  vma->vm_ops = &ops_vm_t;
  vma->vm_private_data = fp->private_data;
  x_vma_open(vma);

  return 0;
}
