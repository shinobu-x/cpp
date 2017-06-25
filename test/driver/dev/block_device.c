#include <linux/fs.h>
#include <linux/genhd.h>
#include <linux/module.h>
#include <linux/kernel.h>

/**
 * Under construction: Should not work
 */

struct bdev_t
{
  int size;                    // Device size in sectors (512*N)
  u8 *data;                    // The data array
  short users;                 // How many users
  short media_change;          // Flag a media change
  spinlock_t lock;             // For mutual exclusion
  struct request_queue *queue; // The device request queue
  struct gendisk *gd;          // The gendisk structure
  struct timer_list timer;     // For simulated media changes
};

static int bdev_open(struct inode* i, struct file* fp) {
  /**
   * private_data:
   * A pointer to block devices internal data
   */
  struct bdev_t* dev = i->i_bdev->bd_disk->private_data;
  del_timer_sync(&dev->timer);
  spin_lock(&dev->lock);

  if (!dev->lock)
    check_disk_change(i->i_bdev);

  dev->users++;
  spin_unlock(&dev->lock);
  return 0;
}

static int bdev_release(struct inode i, struct file* fp) {
  struct bdev_t *dev = i->i_bdev->bd_disk->private_data;
  spin_lock(&dev->lock);
  dev->users--;

  if (!dev->users) {
    dev->timer.expires = jiffies + INVALIDATE_DELAY;
    add_timer(&dev->timer);
  }

  spin_lock(&dev->lock);
  return 0;
}

int bdev_media_changed(struct gendisk *gd) {
  struct bdev_t *dev = gd->private_data;

  return dev->media_change;
}

int bdev_revalidate(struct gendisk *gd) {
  struct bdev_t *dev = gd->private_data;

  if (dev->media_change) {
    dev->media_change = 0;
    memset(dev->data, 0, dev->size);
  }
  return 0;
}

int init_module(void)
{
  return 0;
}

void cleanup_module(void)
{
}
