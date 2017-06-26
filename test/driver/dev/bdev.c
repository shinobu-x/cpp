#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>

#include <linux/kernel.h>   // printk()
#include <linux/fs.h>       // everything
#include <linux/errno.h>    // error codes
#include <linux/types.h>    // size_t
#include <linux/vmalloc.h>
#include <linux/genhd.h>
#include <linux/blkdev.h>
#include <linux/hdreg.h>

MODULE_DESCRIPTION("Block Device Test");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");
static char *Version = "1.0";

static int major_num = 0;
static int sector_size = 512;
static int sectors = 102400;

module_param(major_num, int, 0);
module_param(sector_size, int, 0);
module_param(sectors, int, 0);

#define KERNEL_SECTOR_SIZE 512

static struct request_queue *queue;

static struct bdev_t {
  unsigned long size;
  spinlock_t lock;
  u8 *data;
  struct gendisk *gd;
} Bdev;

/*
 * Handle an I/O request.
 */
static void bdev_transfer(struct bdev_t *dev, sector_t sector,
  unsigned long nsect, char *buffer, int write) {

  unsigned long offset = sectors * sector_size;
  unsigned long nbytes = nsect * sector_size;

  if ((offset + nbytes) > dev->size) {
    printk (KERN_NOTICE "Buffer overflow (%ld %ld)\n", offset, nbytes);
    return;
  }

  write ? memcpy(dev->data + offset, buffer, nbytes)
    : memcpy(buffer, dev->data + offset, nbytes);

  printk(KERN_INFO "bdev: buffer = %c,  offset = %u, len = %u\n",
    buffer, offset, nbytes);
}

static void bdev_request(struct request_queue *q) {

  struct request *req;

  req = blk_fetch_request(q);

  while (req != NULL) {
    // blk_fs_request() was removed in 2.6.36
    //if (!blk_fs_request(req)) {
    if (req == NULL || (req->cmd_type != REQ_TYPE_FS)) {
      printk (KERN_NOTICE "Skip non-CMD request\n");
      __blk_end_request_all(req, -EIO);
      continue;
    }

    bdev_transfer(&Bdev, blk_rq_pos(req), blk_rq_cur_sectors(req), req->buffer,
      rq_data_dir(req)
    );

    if (! __blk_end_request_cur(req, 0))
      req = blk_fetch_request(q);
  }
}

int block_setgeo(struct block_device* block_device, struct hd_geometry * geo) {

  long size;

  size = Bdev.size * (sector_size / KERNEL_SECTOR_SIZE);
  geo->cylinders = (size & ~0x3f) >> 6;
  geo->heads = 4;
  geo->sectors = 16;
  geo->start = 0;
  return 0;
}

/*
 * The device operations structure.
 */
static struct block_device_operations bdev_ops = {
  .owner  = THIS_MODULE,
  .getgeo = block_setgeo
};

static int __init sbd_init(void) {

  Bdev.size = sectors * sector_size;
  spin_lock_init(&Bdev.lock);
  Bdev.data = vmalloc(Bdev.size);

  if (Bdev.data == NULL)
    return -ENOMEM;

  queue = blk_init_queue(bdev_request, &Bdev.lock);

  if (queue == NULL)
    goto out;

  blk_queue_logical_block_size(queue, sector_size);

  major_num = register_blkdev(major_num, "bdev");

  if (major_num < 0) {
    printk(KERN_WARNING "bdev: unable to get major number\n");
    goto out;
  }

  Bdev.gd = alloc_disk(16);

  if (!Bdev.gd)
    goto out_unregister;

  Bdev.gd->major = major_num;
  Bdev.gd->first_minor = 0;
  Bdev.gd->fops = &bdev_ops;
  Bdev.gd->private_data = &Bdev;

  strcpy(Bdev.gd->disk_name, "bdev");
  set_capacity(Bdev.gd, sectors);
  Bdev.gd->queue = queue;
  add_disk(Bdev.gd);

  printk(KERN_INFO "bdev: loaded\n");
  printk(KERN_INFO "major = %d\n", major_num);
  printk(KERN_INFO "buffer size = %d\n", Bdev.size);

  return 0;

out_unregister:
        unregister_blkdev(major_num, "bdev");
out:
  vfree(Bdev.data);
  return -ENOMEM;
}

static void __exit sbd_exit(void)
{
  del_gendisk(Bdev.gd);
  put_disk(Bdev.gd);
  unregister_blkdev(major_num, "bdev");
  blk_cleanup_queue(queue);
  vfree(Bdev.data);
  printk(KERN_INFO "bdev: unloaded\n");
}

module_init(sbd_init);
module_exit(sbd_exit);
