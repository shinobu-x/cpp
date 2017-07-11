#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>
#include <linux/kernel.h>  /** container_of **/
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/fcntl.h>
#include <linux/aio.h>
#include <asm/uaccess.h>

#include "common.h"

MODULE_AUTHOR("Shinobu Kinjo");
MODULE_DESCRIPTION("Memory Mapping Test part1");
MODULE_LICENSE("GPL");

int x_major = X_MAJOR;
int x_devs = X_DEVS;
int x_qset = X_QSET;
int x_order = X_ORDER;

module_param(x_major, int, 0);
module_param(x_devs, int, 0);
module_param(x_qset, int, 0);
module_param(x_order, int, 0);

struct x_dev_t* devs_t;
int x_trim(struct x_dev_t* dev);
void x_cleanup(void);

#ifdef USE_PROC

int do_read_proc(struct seq_file* m, void* v) {
  int i, j, order, qset;
  int limit = m->size - 80;
  struct x_dev_t* dev;

  for (i=0; i<x_devs; ++i) {
    dev = &devs_t[i];

    if (down_interruptible(&dev->sem))
      return -ERESTARTSYS;

    qset = dev->qset;
    order = dev->order;
    seq_printf(m, "\ndevice %i: qset %i, order %i, sz %li\n",
      i, qset, order, (long)(dev->size));

    for (; dev; dev = dev->next) {
      seq_printf(m, " item at %p, qset at %p\n", dev, dev->data);

      if (m->count > limit)
        goto out;

      if (dev->data && !dev->next)
        for (j=0; j<qset; ++j) {

          if (dev->data[j])
            seq_printf(m, " %4i:%8p\n",j, dev->data[j]);

          if (m->count > limit)
            goto out;
        }
     }
out:
  up(&devs_t[i].sem);

  if (m->count > limit)
    break;
  }
  return 0;
}

static int do_open_proc(struct inode* inode, struct file* fp) {
  return single_open(fp, do_read_proc, NULL);
}

static struct file_operations ops_proc_t = {
  .owner = THIS_MODULE,
  .open = do_open_proc,
  .read = seq_read,
  .llseek = seq_lseek,
  .release = single_release
};

#endif // USE_PROC

int x_open(struct inode* inode, struct file* fp) {
  struct x_dev_t* dev;

  dev = container_of(inode->i_cdev, struct x_dev_t, cdev);

  if ((fp->f_flags & O_ACCMODE) == O_WRONLY) {
    if (down_interruptible(&dev->sem))
      return -ERESTARTSYS;
    x_trim(dev);
    up(&dev->sem);
  }

  // Pointing to the device data
  fp->private_data = dev;

  return 0;
}

int x_release(struct inode* inode, struct file* fp) {
  return 0;
}

struct x_dev_t* x_follow(struct x_dev_t* dev, int n) {
  while (n--) {
    if (!dev->next) {
      dev->next = kmalloc(sizeof(struct x_dev_t), GFP_KERNEL);
      memset(dev->next, 0, sizeof(struct x_dev_t));
    }

    dev = dev->next;
    continue;
  }
  return dev;
}

ssize_t x_read(struct file* fp, char* __user buf, size_t count, loff_t* f_pos) {
  struct x_dev_t* dev = fp->private_data;
  struct x_dev_t* ptr;
  int quantum = PAGE_SIZE << dev->order;
  int qset = dev->qset;
  int itemsize = quantum*qset;
  int item, s_pos, q_pos, rest;
  ssize_t r = 0;

  if (down_interruptible(&dev->sem))
    return -ERESTARTSYS;
  if (*f_pos > dev->size)
    goto do_nothing;
  if (*f_pos + count > dev->size)
    count = dev->size - *f_pos;

  item = ((long)* f_pos)/itemsize;
  rest = ((long)* f_pos)%itemsize;
  s_pos = rest/quantum;
  q_pos = rest%quantum;

  ptr = x_follow(dev, item);

  if (!ptr->data)
    goto do_nothing;
  if (!ptr->data[s_pos])
    goto do_nothing;
  if (count > quantum - q_pos)
    count = quantum - q_pos;

  if (copy_to_user(buf, ptr->data[s_pos]+q_pos, count)) {
    r = -EFAULT;
    goto do_nothing;
  }

  up(&dev->sem);

  *f_pos += count;
  return count;

do_nothing:
  up(&dev->sem);
  return r;
}

ssize_t x_write(struct file* fp, const char* __user buf, size_t count,
  loff_t* f_pos) {
  struct x_dev_t* dev = fp->private_data;
  struct x_dev_t* ptr;
  int quantum = PAGE_SIZE << dev->order;
  int qset = dev->qset;
  int itemsize = quantum*qset;
  int item, s_pos, q_pos, rest;
  ssize_t r = -ENOMEM;

  if (down_interruptible(&dev->sem))
    return -ERESTARTSYS;

  item = ((long) *f_pos)/itemsize;
  rest = ((long) *f_pos)%itemsize;
  s_pos = rest/quantum;
  q_pos = rest%quantum;

  ptr = x_follow(dev, item);
  if (!ptr->data) {
    ptr->data = kmalloc(qset*sizeof(void*), GFP_KERNEL);

    if (!ptr->data)
      goto nomem;

    memset(ptr->data, 0, qset*sizeof(char*));

  }

  if (count > quantum - q_pos)
    count = quantum - q_pos;

  if (copy_from_user(ptr->data[s_pos]+q_pos, buf, count)) {
    r = -EFAULT;
    goto nomem;
  }

  *f_pos += count;

  if (dev->size < *f_pos)
    dev->size = *f_pos;

  up(&dev->sem);

  return count;

nomem:
  up(&dev->sem);

  return r;
}

long x_ioctl(struct file* fp, unsigned int cmd, unsigned long arg) {
  int e = 0, r = 0, t;

  if (_IOC_TYPE(cmd) != X_IOC_MAGIC)
    return -ENOTTY;

  if (_IOC_NR(cmd) > X_IOC_MAXNR)
    return -ENOTTY;

  if (_IOC_DIR(cmd) & _IOC_READ)
    e = !access_ok(VERIFY_WRITE, (void* __user)arg, _IOC_SIZE(cmd));
  else if (_IOC_DIR(cmd) & _IOC_WRITE)
    e = !access_ok(VERIFY_READ, (void* __user)arg, _IOC_SIZE(cmd));

  if (e)
    return -EFAULT;

  switch (cmd) {

  case X_IOCRESET:
    x_qset = X_QSET;
    x_order = X_ORDER;
    break;
  case X_IOCSORDER:
    x_order = arg;
    break;
  case X_IOCGORDER:
    r = __put_user(x_order, (int* __user)arg);
    break;
  case X_IOCQORDER:
    return x_order;
  case X_IOCXORDER:
    t = x_order;
    r = __get_user(x_order, (int __user*)arg);
    if (r == 0)
      r = __put_user(t, (int __user*)arg);
    break;
  case X_IOCHORDER:
    t = x_order;
    x_order = arg;
    return t;
  default:
    return -ENOTTY;
  }

  return r;
}

loff_t x_llseek(struct file* fp, loff_t off, int there) {
  struct x_dev_t* dev = fp->private_data;
  long newpos;

  switch (there) {
  case 0:
    newpos = off;
    break;

  case 1:
    newpos = fp->f_pos + off;
    break;

  case 2:
    newpos = dev->size + off;
    break;

  default:
    return -EINVAL;
  }

  if (newpos < 0)
    return -EINVAL;

  fp->f_pos = newpos;

  return newpos;
}

struct delay_work_t {
  struct kiocb* iocb;
  int r;
  struct delayed_work dw;
};

static void x_do_deferred_op(struct work_struct* w) {
  struct delay_work_t* stuff = container_of(w, struct delay_work_t, dw.work);
  aio_complete(stuff->iocb, stuff->r, 0);
  kfree(stuff);
}

static int x_defer_op(int is_write, struct kiocb* iocb,
  const struct iovec* iovec,
  unsigned long nr_segs, loff_t pos) {

  struct delay_work_t* stuff;
  int r = 0;
  size_t l = 0;
  unsigned long s;

  for (s = 0; s < nr_segs; ++s) {
    if (is_write)
      l = x_write(iocb->ki_filp, iovec[s].iov_base, iovec[s].iov_len, &pos);
    else
      l = x_read(iocb->ki_filp, iovec[s].iov_base, iovec[s].iov_len, &pos);

    if (l < 0)
      return l;

    r += l;
  }

  // Synchronous IOCB?
  if (is_sync_kiocb(iocb))
    return r;

  stuff = kmalloc(sizeof(*stuff), GFP_KERNEL);

  if (stuff == NULL)
    return r;

  stuff->iocb = iocb;
  stuff->r = r;
  INIT_DELAYED_WORK(&stuff->dw, x_do_deferred_op);
  schedule_delayed_work(&stuff->dw, HZ/100);

  return -EIOCBQUEUED;
}

static ssize_t x_aio_read(struct kiocb* iocb, const struct iovec* iovec,
  unsigned long nr_segs, loff_t pos) {
  return x_defer_op(0, iocb, iovec, nr_segs, pos);
}

static ssize_t x_aio_write(struct kiocb* iocb, const struct iovec* iovec,
  unsigned long nr_segs, loff_t pos) {
  return x_defer_op(1, iocb, iovec, nr_segs, pos);
}

extern int x_mmap(struct file* fp, struct vm_area_struct* vma);

struct file_operations ops_t = {
  .owner = THIS_MODULE,
  .llseek = x_llseek,
  .read = x_read,
  .write = x_write,
  .unlocked_ioctl = x_ioctl,
  .mmap = x_mmap,
  .open = x_open,
  .release = x_release,
  .aio_read = x_aio_read,
  .aio_write = x_aio_write,
};

int x_trim(struct x_dev_t* dev) {
  struct x_dev_t* next;
  struct x_dev_t* ptr;
  int i;

  // Active mapping?
  if (dev->vmas)
    return -EBUSY;

  for (ptr = dev; ptr; ptr = next) {
    if (ptr->data) {
      for (i = 0; i < x_qset; ++i)
        if (ptr->data[i])
          free_pages((unsigned long)(ptr->data[i]), ptr->order);

      kfree(ptr->data);
      ptr->data = NULL;
    }
    next = ptr->next;

    if (ptr != dev)
      kfree(ptr);
  }

  dev->size = 0;
  dev->qset = x_qset;
  dev->order = x_order;
  dev->next = NULL;

  return 0;
}

static void x_setup_cdev(struct x_dev_t* dev, int idx) {
  int e;
  int n = MKDEV(x_major, idx);
  cdev_init(&dev->cdev, &ops_t);
  dev->cdev.owner = THIS_MODULE;
  dev->cdev.ops = &ops_t;
  e = cdev_add(&dev->cdev, n, 1);

  if (e)
    printk(KERN_NOTICE "Error: %d, device id %d", e, idx);
}

static int __init do_setup(void) {
  int r, i;
  dev_t dev = MKDEV(x_major, 0);

  if (x_major)
    r = register_chrdev_region(dev, x_devs, "xp");
  else {
    r = alloc_chrdev_region(&dev, 0, x_devs, "xp");
    x_major = MAJOR(dev);
  }

  if (r < 0)
    return r;

  devs_t = kmalloc(x_devs*sizeof(struct x_dev_t), GFP_KERNEL);

  if (!devs_t) {
    r = -ENOMEM;
    goto nomem;
  }

  memset(devs_t, 0, x_devs*sizeof(struct x_dev_t));

  for (i=0; i<x_devs; ++i) {
    devs_t[i].order = x_order;
    devs_t[i].qset = x_qset;
    sema_init(&devs_t[i].sem, 1);
    x_setup_cdev(devs_t + i, i);
  }

#ifdef USE_PROC
  proc_create("xp", 0, NULL, &ops_proc_t);
#endif

  return 0;

nomem:
  unregister_chrdev_region(dev, x_devs);
  return r;
}

static void __exit do_cleanup(void) {
  int i;

#ifdef USE_PROC
  remove_proc_entry("xp", NULL);
#endif

  for (i=0; i<x_devs; i++) {
    cdev_del(&devs_t[i].cdev);
    x_trim(devs_t + i);
  }

  kfree(devs_t);
  unregister_chrdev_region(MKDEV(x_major, 0), x_devs);
}

module_init(do_setup);
module_exit(do_cleanup);
