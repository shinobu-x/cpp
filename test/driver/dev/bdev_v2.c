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

MODULE_DESCRIPTION("Block Device Test Part2");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

#define KERNEL_SECTOR_SIZE 512

static int major_number = 0;
static int minor_number = 16;
static int sector_size = 512;
static int sectors = 102400;

module_param(major_number, int, 0);
module_param(sector_size, int, 0);
module_param(sectors, int, 0);

static struct request_queue *queue;

static struct bdev_t {
  unsigned long size;
  spinlock_t lock;
  u8 *data;
  struct gendisk *gd;
} Bdev;

/**
 * I/O operations
 */
static void transfer(struct bdev_t *dev, sector_t sector,
  unsigned long nsect, char *buffer, int write) {

  unsigned long offset = sector * sector_size;  /* Object start byte */
  unsigned long nbytes = nsect * sector_size;   /* Bytes from offset */

  if ((offset + nbytes) > dev->size) {
    printk(KERN_NOTICE "bdev: Buffer overflow (%ld %ld)\n", offset, nbytes);
    return;
  }

  write ? memcpy(dev->data + offset, buffer, nbytes) :
    memcpy(buffer, dev->data + offset, nbytes);

  printk(KERN_INFO "bdev: offset = %lu, len = %lu\n", offset, nbytes);
}

/**
 * I/O request
 */
static void request(struct request_queue *q) {

  struct request *req;

  req = blk_fetch_request(q);

  while (req != NULL) {
    if (req == NULL || (req->cmd_type != REQ_TYPE_FS)) {
      printk(KERN_NOTICE "bdev: Skip command request\n");
      __blk_end_request_all(req, -EIO);
      continue;
    }

    /**
     * blk_rq_pos:          the current sector
     * blk_rq_cur_sectors:  sectors left in the current segment
     */
    transfer(&Bdev, blk_rq_pos(req), blk_rq_cur_sectors(req), req->buffer,
      rq_data_dir(req));

    if (!__blk_end_request_cur(req, 0))
      req = blk_fetch_request(q);
  }
}

int setgeo(struct block_device* block_device, struct hd_geometry* geo) {
  long size;
  size = Bdev.size * (sector_size / KERNEL_SECTOR_SIZE);
  geo->cylinders = (size & ~0x3f) >> 6;
  geo->heads = 4;
  geo->sectors = 16;
  geo->start = 0;
  return 0;
}

static struct block_device_operations bdev_ops = {
  .owner = THIS_MODULE,
  .getgeo = setgeo
};

static int __init bdev_init(void) {
  Bdev.size = sectors * sector_size;
  spin_lock_init(&Bdev.lock);
  Bdev.data = vmalloc(Bdev.size);

  if (Bdev.data == NULL)
    return -ENOMEM;

  queue = blk_init_queue(request, &Bdev.lock);

  if (queue == NULL)
    goto out;

  blk_queue_logical_block_size(queue, sector_size);

  major_number = register_blkdev(major_number, "bdev");

  if (major_number < 0) {
    printk(KERN_WARNING "bdev: Unable to get major number\n");
    goto out;
  }

  Bdev.gd = alloc_disk(minor_number);

  if (!Bdev.gd)
    goto out_unregister;

  Bdev.gd->major = major_number;
  Bdev.gd->first_minor = 0;
  Bdev.gd->fops = &bdev_ops;

  strcpy(Bdev.gd->disk_name, "bdev");
  set_capacity(Bdev.gd, sectors);
  Bdev.gd->queue = queue;
  /**
   * Add partitioning information to kernel list
   * @disk: per-device partitioning information
   */
  add_disk(Bdev.gd);

  printk(KERN_INFO "bdev: loaded\n");
  printk(KERN_INFO "bdev: major = %d, minor = %d\n",
    major_number, minor_number);
  printk(KERN_INFO "bdev: buffer size = %lu\n", Bdev.size);

  return 0;

out_unregister:
  unregister_blkdev(major_number, "bdev");

out:
  vfree(Bdev.data);
  return -ENOMEM;
}

static void __exit bdev_exit(void) {
  del_gendisk(Bdev.gd);
  put_disk(Bdev.gd);
  unregister_blkdev(major_number, "bdev");
  blk_cleanup_queue(queue);
  vfree(Bdev.data);
  printk(KERN_INFO, "bdev: unloaded\n");
}

module_init(bdev_init);
module_exit(bdev_exit);
