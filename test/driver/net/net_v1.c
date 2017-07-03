#include <linux/etherdevice.h>
#include <linux/errno.h>
#include <linux/in.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/ip.h>
#include <linux/in6.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/netdevice.h>
#include <linux/sched.h>
#include <linux/skbuff.h>
#include <linux/slab.h>
#include <linux/tcp.h>
#include <linux/types.h>

#include <asm/checksum.h>

MODULE_DESCRIPTION("Net Dev Test Part1");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

#define MY_RX_INTR 0x0001
#define MY_TX_INTR 0x0002
#define MY_TIMEOUT 5

static int lockup = 0;
static int timeout = MY_TIMEOUT;
static int use_napi = 0;
int pool_size = 8;

module_param(lockup, int, 0);
module_param(timeout, int, 0);
module_param(use_napi, int, 0);
module_param(pool_size, int, 0);

// Structure for an in-flight packet
struct my_packet {
  struct my_packet* next;
  struct net_device* dev;
  int len;
  u8 data[ETH_DATA_LEN];
};

// Private data for each device.
struct my_priv {
  struct net_device_stats stats;
  int status;
  struct my_packet *ppool;
  struct my_packet *rx_queue;  // Incoming packets
  int rx_int_enable;
  int tx_packetlen;
  u8* tx_packetdata;
  struct sk_buff* skb;
  spinlock_t lock;
  struct net_device* dev;
  struct napi_struct napi;
};

static void my_tx_timeout(struct net_device* dev);
static void (*my_interrupt)(int, void*, struct pt_regs*);

// Set up packet pool for each device
void setup_pool(struct net_device* dev) {
  struct my_priv* priv = netdev_priv(dev);
  int i;
  struct my_packet* pkt;

  priv->ppool = NULL;

  for (i=0; i<pool_size; i++) {
    pkt = kmalloc(sizeof(struct my_packet), GFP_KERNEL);

    if (pkt==NULL) {
      printk(KERN_NOTICE "Overflow\n");
      return;
    }

    pkt->dev = dev;
    pkt->next = priv->ppool;
    priv->ppool = pkt;
  }
}

void teardown_pool(struct net_device* dev) {
  struct my_priv* priv = netdev_priv(dev);
  struct my_packet* pkt;

  while ((pkt = priv->ppool)) {
    priv->ppool = pkt->next;
    kfree(pkt);
  }
}

struct my_packet* get_tx_buffer(struct net_device* dev) {
  struct my_priv* priv = netdev_priv(dev);
  unsigned long flags;
  struct my_packet* pkt;

  spin_lock_irqsave(&priv->lock, flags);
  pkt = priv->ppool;
  priv->ppool = pkt->next;

  if (priv->ppool == NULL) {
    printk(KERN_INFO "Empty pool\n");
    netif_stop_queue(dev);
  }

  spin_unlock_irqrestore(&priv->lock, flags);
  return pkt;
}

void release_buffer(struct my_packet* pkt) {
  unsigned long flags;
  struct my_priv* priv = netdev_priv(pkt->dev);

  spin_lock_irqsave(&priv->lock, flags);
  pkt->next = priv->ppool;
  priv->ppool = pkt;
  spin_unlock_irqrestore(&priv->lock, flags);

  if (netif_queue_stopped(pkt->dev) && pkt->next == NULL)
    netif_wake_queue(pkt->dev);
}

void enqueue_buf(struct net_device* dev, struct my_packet* pkt) {
  unsigned long flags;
  struct my_priv* priv = netdev_priv(dev);

  spin_lock_irqsave(&priv->lock, flags);
  pkt->next = priv->rx_queue;
  priv->rx_queue = pkt;
  spin_unlock_irqrestore(&priv->lock, flags);
}

struct my_packet* dequeue_buf(struct net_device* dev) {
  struct my_priv* priv = netdev_priv(dev);
  struct my_packet* pkt;
  unsigned long flags;

  spin_lock_irqsave(&priv->lock, flags);
  pkt = priv->rx_queue;

  if (pkt != NULL)
    priv->rx_queue = pkt->next;

  spin_unlock_irqrestore(&priv->lock, flags);

  return pkt;
}
