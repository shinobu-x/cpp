#include <linux/etherdevice.h>  // eth_type
#include <linux/errno.h>        // Error codes
#include <linux/in.h>
#include <linux/init.h>
#include <linux/interrupt.h>    // mark_bh
#include <linux/ip.h>           // struct iphdr
#include <linux/in6.h>
#include <linux/kernel.h>       // printk
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/netdevice.h>    // struct device, and more
#include <linux/sched.h>
#include <linux/skbuff.h>
#include <linux/slab.h>         // kmalloc
#include <linux/tcp.h>          // struct tcphdr
#include <linux/types.h>        // size_t

#include <asm/checksum.h>

MODULE_DESCRIPTION("Net Dev Test Part2");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

#define RX_INTR 0x0001
#define TX_INTR 0x0002
#define TIMEOUT 5

struct net_device* net_devs[2];

static int lockup = 0;
static int timeout = TIMEOUT;
static int use_napi = 0;
int pool_size = 8;

module_param(lockup, int, 0);
module_param(timeout, int, 0);
module_param(use_napi, int, 0);
module_param(pool_size, int, 0);

// Structure for an in-flight packet
struct packet_t {
  struct packet_t* next;
  struct net_device* dev;
  int data_len;
  u8 data[ETH_DATA_LEN];
};

// Private data for each device
struct priv_t {
  struct net_device_stats stats;
  int status;
  struct packet_t* packet_pool;
  struct packet_t* rx_queue;
  int rx_int_enabled;
  int tx_packet_len;
  u8* tx_packet_data;
  struct sk_buff* skb;
  spinlock_t l;
  struct net_device* dev;
  struct napi_struct napi;
};

static void tx_timeout(struct net_device* dev);
static void (*interrup)(int, void*, struct pt_regs*);

void setup_pool(struct net_device* dev) {
  struct priv_t* priv = netdev_priv(dev);
  struct packet_t* pkt;
  int i;
  priv->packet_pool = NULL;

  for (i=0; i<pool_size; ++i) {

    pkt = kmalloc(sizeof(struct packet_t), GFP_KERNEL);

    if (pkt == NULL) {
      printk(KERN_WARNING "net_v2: Overflow\n");
      return;
    }

    pkt->dev = dev;
    pkt->next = priv;
    priv->packet_pool = pkt;
  }
}

void do_teardown_pool(struct net_device* dev) {
  struct priv_t* priv = netdev_priv(dev);
  struct packet_t* pkt;

  while (pkt = priv->packet_pool) {
    priv->packet_pool = pkt->next;
    kfree(pkt);
  }
}

struct packet_t* get_tx_buffer(struct net_device* dev) {
  struct priv_t* priv = netdev_priv(dev);
  unsigned long flags;
  struct packet_t* pkt;

  // Get lock and disable cpu interruption
  spin_lock_irqsave(&priv->l, flags);
  pkt = priv->packet_pool;
  priv->packet_pool = pkt->next;

  if (priv->packet_pool == NULL) {
    printk(KERN_INFO "net_v2: Pool is empty\n");
    netif_stop_queue(dev);
  }

  // Release lock and enable cpu interruption
  spin_unlock_irqrestore(&priv->l, flags);

  return pkt;
}

void release_buffer(struct packet_t* pkt) {
  unsigned long flags;
  struct priv_t* priv = netdev_priv(pkt->dev);

  spin_lock_irqsave(&priv->l, flags);
  pkt->next = priv->packet_pool;
  priv->packet_pool = pkt;
  spin_unlock_irqrestore(&priv->l, flags);

  if (netif_queue_stopped(pkt->dev) && pkt->next == NULL)
    // Start transmitting packets again
    netif_wake_queue(pkt->dev);
}

void enqueue_buffer(struct net_device* dev, struct packet_t* pkt) {
  unsigned long flags;
  struct priv_t* priv = netdev_priv(dev);

  spin_lock_irqsave(&priv->l, flags);
  pkt->next = priv->packet_pool;
  priv->rx_queue = pkt;
  spin_unlock_irqrestore(&priv->l, flags);
}

struct packet_t* dequeue_buffer(struct net_device* dev) {
  struct priv_t* priv = netdev_priv(dev);
  struct packet_t* pkt;
  unsigned long flags;

  spin_lock_irqsave(&priv->l, flags);
  pkt = priv->rx_queue;

  if (pkt != NULL)
    priv->rx_queue = pkt->next;

  spin_unlock_irqrestore(&priv->l, flags);

  return pkt;
}

struct packet_t* rx_ints(struct net_device* dev, int enable) {
  struct priv_t* priv = netdev_priv(dev);
  priv->rx_int_enabled = enable;
}

int do_open(struct net_device* dev) {
  memcpy(dev->dev_addr, "\0TEST_IF0", ETH_ALEN);

  if (dev == net_devs[1])
    dev->dev_addr[ETH_ALEN-1]++;

  netif_start_queue(dev);

  return 0;
}

int do_release(struct net_device* dev) {
  netif_stop_queue(dev);
  return 0;
}

int do_config(struct net_device* dev, struct ifmap* map) {
  if (dev->flags & IFF_UP)
    return -EBUSY;

  if (map->base_addr != dev->base_addr) {
    printk(KERN_WARNING "net_v2: Failed to change address\n");
    return -EOPNOTSUPP;
  }

  if (map->irq != dev->irq)
    dev->irq = map->irq;

  return 0;
}

void handle_rx(struct net_device* dev, struct packet_t* pkt) {
  struct sk_buff* skb;
  struct priv_t* priv = netdev_priv(dev);

  skb = dev_alloc_skb(pkt->data_len + 2);

  if (!skb) {
    if (printk_ratelimit())
      printk(KERN_NOTICE "net_v2: RX: Low on memory - packet dropped\n");

    priv->stats.rx_dropped++;
    goto out;
  }

  skb_reserve(skb, 2);
  memcpy(skb_put(skb, pkt->data_len), pkt->data, pkt->data_len);

  skb->dev = dev;
  skb->protocol = eth_type_trans(skb, dev);
  skb->ip_summed = CHECKSUM_UNNECESSARY;

  priv->stats.rx_packets++;
  priv->stats.rx_bytes += pkt->data_len;
  // Notify the kernel that a packet has been received and ecapsulated into a s-
  // ocket buffer
  netif_rx(skb);

out:
  return;
}

static int do_poll(struct napi_struct* napi, int budget) {
  int n_packets = 0;
  struct sk_buff* skb;
  struct priv_t* priv = container_of(napi, struct priv_t, napi);
  struct net_device* dev = priv->dev;
  struct packet_t* pkt;

  while (n_packets < budget && priv->rx_queue) {
    pkt = dequeue_buffer(dev);
    skb = dev_alloc_skb(pkt->data_len + 2);

    if (!skb) {
      if (printk_ratelimit())
        printk(KERN_NOTICE "net_v2: Packet is dropped\n");

      priv->stats.rx_dropped++;
      release_buffer(pkt);

      continue;
    }

    skb_reserve(skb, 2);
    memcpy(skb_put(skb, pkt->data_len), pkt->data, pkt->data_len);
    skb->dev = dev;
    skb->ip_summed = CHECKSUM_UNNECESSARY;
    netif_receive_skb(skb);

    n_packets++;
    priv->stats.rx_packets++;
    priv->stats.rx_bytes += pkt->data_len;
    release_buffer(pkt);
  }

  if (!priv->rx_queue) {
    napi_complete(napi);
    rx_ints(dev, 1);
    return 0;
  }

  return n_packets;
}

/**
 * < struct pt_regs > defines the way the registers are stored on the stack dur-
 * ng a system call
 */
static void regular_interrupt(int irq, void* dev_id, struct pt_regs* regs) {
  int s; // Status
  struct priv_t* priv;
  struct packet_t* pkt = NULL;
  struct net_device* dev = (struct net_device*)dev_id;

  if (!dev) return; // ...

  priv = netdev_priv(dev);
  spin_lock(&priv->l);

  s = priv->status;
  priv->status = 0;

  if (s & RX_INTR) {
    priv->stats.tx_packets++;
    priv->stats.tx_bytes += priv->tx_packet_len;
    dev_kfree_skb(priv->skb);
  }

  spin_unlock(&priv->l);

  if (pkt)
    release_buffer(pkt);

  return;
}

static void napi_interrupt(int irq, void* dev_id, struct pt_regs* regs) {
  int s; // Status
  struct priv_t* priv;
  struct net_device* dev = (struct net_device*)dev_id;

  if (!dev) return; // ...

  priv = netdev_priv(dev);
  spin_lock(&priv->l);

  s = priv->status;
  priv->status = 0;

  if (s & RX_INTR) {
    rx_ints(dev, 0);
    napi_schedule(&priv->napi);
  }

  if (s & TX_INTR) {
    priv->stats.tx_packets++;
    priv->stats.tx_bytes += priv->tx_packet_len;
  }

  spin_unlock(&priv->l);

  return;
}

static void hw_tx(char* buf, int len, struct net_device* dev) {
  struct iphdr* header;
  struct net_device* dest;
  struct priv_t* priv;
  u32* saddr, daddr;
  struct packet_t* tx_buf;

  if (len < sizeof(struct ethhdr) + sizeof(struct iphdr)) {
    printk("net_v2: %i octet\n", size);
    return;
  }

  if (0) {
    int i;
    printk(KERN_NOTICE "Length: %i\n" KERN_DEBUG " data: ", len);

    for (i=14; i<len; ++i)
      printk(" %02x", buf[i]&0xff);
    printk("\n");
  }

  header = (struct iphdr*)(buf+sizeof(struct ethhdr));
  saddr = &header->saddr;
  daddr = &header->daddr;

  ((u8*)saddr)[2] ^= 1;
  ((u8*)daddr)[2] ^= 1;

  header->check = 0;
  header->check = ip_fast_csum((unsigned char*)header, header->ihl);
