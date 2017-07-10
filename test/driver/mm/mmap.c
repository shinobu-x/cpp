#include <linux/module.h>
#include <linux/mm.h>
#include <asm/pgtable.h>
#include <linux/fs.h>

#include "common.h"

#define X_MAJOR 0
#define X_DEVS 4
#define X_ORDER 0
#define X_QSET 500

struct dev_t {
  void** data;
  struct dev_t* next;
  int vmas;
  int order;
  int qset;
  size_t size;
  struct semaphore sem;
  struct cdev cdev;
};

extern struct dev_t* devs_t;
extern struct file_operations op_t;
extern int x_major;
extern int x_devs;
extern int x_order;
extern int x_qset;

int x_trim(struct dev_t* dev)
struct dev_t* x_follow(struct dev_t* dev, int n);

#define X_IOC_MAGIC 'K'
#define X_IOC_RESET _IO(IOC_MAGIC, 0)

/**
 * S: Set through a ptr
 * T: Tell directly
 * G: Get (to apointed var)
 * Q: Query, response is on the return value
 * X: Exchange G and S atomically
 * H: Shift T and Q atomically
 */

#define X_IOCSORDER   _IOW(X_IOC_MAGIC,  1, int)
#define X_IOCTORDER   _IO(X_IOC_MAGIC,   2)
#define X_IOCGORDER   _IOR(X_IOC_MAGIC,  3, int)
#define X_IOCQORDER   _IO(X_IOC_MAGIC,   4)
#define X_IOCXORDER   _IOWR(X_IOC_MAGIC, 5, int)
#define X_IOCHORDER   _IO(X_IOC_MAGIC,   6)
#define X_IOCSQSET    _IOW(X_IOC_MAGIC,  7, int)
#define X_IOCTQSET    _IO(X_IOC_MAGIC,   8)
#define X_IOCGQSET    _IOR(X_IOC_MAGIC,  9, int)
#define X_IOCQQSET    _IO(X_IOC_MAGIC,  10)
#define X_IOCXQSET    _IOWR(X_IOC_MAGIC,11, int)
#define X_IOCHQSET    _IO(X_IOC_MAGIC,  12)

#define X_IOC_MAXNR 12

void x_vma_open(struct vma_area_struct* vma) {
  struct dev_t* dev = vma->vm_private_data;
  dev->vmas++;
}

void x_vma_close(struct vma_area_struct* vma) {
  struct dev_t* dev = vma_vm_private_data;
  dev->vmas--;
}
