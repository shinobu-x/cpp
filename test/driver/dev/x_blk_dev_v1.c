#include <linux/blkdev.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/genhd.h>
#include <linux/hdreg.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/types.h>
#include <linux/vmalloc.h>

MODULE_DESCRIPTION("Block Device Test");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

#define KERNEL_SECTOR_SIZE 512

static int major = 0;
static int minor = 16;
static int size_of_sector = 512;
static int number_of_sector = 1024000;

static struct request_queue* queue;

static struct x_blk_dev_t {
  unsigned long size;
  spinlock_t l;
  u8* data;
  struct gendisk* gd;
} X_BLK_DEV_T;

static void do_transfer(struct x_blk_dev_t*, sector_t sector,
  unsigned long nsec, char* buffer, int is_write);
static void do_request(struct request_queue* que);
int setgeo(struct block_device* block_device, struct hd_geometry* geo);

static void do_transfer(struct x_blk_dev_t* dev,
  sector_t sector, unsigned long nsec, char* buffer, int is_write) {

  unsigned long offset = sector * size_of_sector;  /* Object start byte */
  unsigned long bytes = nsec * size_of_sector;     /* Bytes from offset */

  if ((offset + bytes) > dev->size) {
    printk(KERN_WARNING "x_blk_dev: Buffer overflow, offset=%ld, bytes=%ld | size=%ld\n",
      offset, bytes, dev->size);
    return;
  }

  is_write ? memcpy(dev->data+offset, buffer, bytes) :
    memcpy(buffer, dev->data+offset, bytes);

  printk(KERN_INFO "x_blk_dev: offset=%lu, length=%lu\n", offset, bytes);
}

static void do_request(struct request_queue* que) {

  struct request* req;
  req = blk_fetch_request(que);

  while (req != NULL) {
    if (req == NULL || (req->cmd_type != REQ_TYPE_FS)) {

      printk(KERN_WARNING "x_blk_dev: Ignore requested command\n");
      __blk_end_request_all(req, -EIO);
      continue;
    }
    /**
     * blk_rq_pos:         The current sector
     * blk_rq_cur_sectors: Sectors left in the current segment
     **/
    do_transfer(&X_BLK_DEV_T, blk_rq_pos(req), blk_rq_cur_sectors(req),
      req->buffer, rq_data_dir(req));

    if (!__blk_end_request_cur(req, 0))
      req = blk_fetch_request(que);
  }
}

int setgeo(struct block_device* block_device, struct hd_geometry* geo) {

  long size;
  size = X_BLK_DEV_T.size*(size_of_sector/KERNEL_SECTOR_SIZE);

  geo->cylinders = (size & ~0x3f) >> 6;
  geo->heads = 4;
  geo->sectors = 16;
  geo->start = 0;

  return 0;
}

static struct block_device_operations op_t = {

  .owner = THIS_MODULE,
  .getgeo = setgeo
};

static int __init do_init(void) {

  X_BLK_DEV_T.size = number_of_sector*size_of_sector;
  spin_lock_init(&X_BLK_DEV_T.l);
  X_BLK_DEV_T.data = vmalloc(X_BLK_DEV_T.size);

  if (X_BLK_DEV_T.data == NULL)
    return -ENOMEM;

  queue = blk_init_queue(do_request, &X_BLK_DEV_T.l);

  if (queue == NULL)
    goto out;

  blk_queue_logical_block_size(queue, size_of_sector);

  major = register_blkdev(major, "x_blk_dev");

  if (major < 0) {
    printk(KERN_WARNING "x_blk_dev: Unable to get major number\n");
    goto out;
  }

  X_BLK_DEV_T.gd = alloc_disk(minor);

  if (!X_BLK_DEV_T.gd)
    goto do_unregister;

  X_BLK_DEV_T.gd->major = major;
  X_BLK_DEV_T.gd->first_minor = 0;
  X_BLK_DEV_T.gd->fops = &op_t;

  strcpy(X_BLK_DEV_T.gd->disk_name, "x_blk_dev");
  set_capacity(X_BLK_DEV_T.gd, number_of_sector);
  X_BLK_DEV_T.gd->queue = queue;

  /**
   * Add partitioning information to kernel list.
   * @disk: per-device partitioning information.
   **/
  add_disk(X_BLK_DEV_T.gd);

  printk(KERN_INFO "x_blk_dev: Module loaded\n");
  printk(KERN_INFO "x_blk_dev: major=%d, minor=%d\n", major, minor);
  printk(KERN_INFO "x_blk-dev: buffer size=%lu\n", X_BLK_DEV_T.size);

  return 0;

do_unregister:
  unregister_blkdev(major, "x_blk_dev");

out:
  vfree(X_BLK_DEV_T.data);
  return -ENOMEM;
}

static void __exit do_cleanup(void) {
  del_gendisk(X_BLK_DEV_T.gd);
  put_disk(X_BLK_DEV_T.gd);
  unregister_blkdev(major, "x_blk_dev");
  vfree(X_BLK_DEV_T.data);
  printk(KERN_INFO "x_blk_dev: Module unloaded\n");
}

module_init(do_init);
module_exit(do_cleanup);
