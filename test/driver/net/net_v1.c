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

struct net_device* my_devs[2];

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
  int data_size;
  u8 data[ETH_DATA_LEN];
};

// Private data for each device.
struct my_priv {
  struct net_device_stats stats;
  int status;
  struct my_packet *ppool;
  struct my_packet *rx_queue;  // Incoming packets
  int rx_int_enabled;
  int tx_packet_size;
  u8* tx_packet_data;
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

static void rx_ints(struct net_device* dev, int enable) {
  struct my_priv* priv = netdev_priv(dev);
  priv->rx_int_enabled = enable;
}

int do_open(struct net_device* dev) {
  memcpy(dev->dev_addr, "\0MYIF0", ETH_ALEN);

  if (dev == my_devs[1])
    dev->dev_addr[ETH_ALEN-1]++;

  netif_start_queue(dev);

  return 0;
}

int do_release(struct net_device* dev) {
  netif_stop_queue(dev);
  return 0;
}

int my_config(struct net_device* dev, struct ifmap* map) {
  if (dev->flags & IFF_UP)
    return -EBUSY;

  if (map->base_addr != dev->base_addr) {
    printk(KERN_WARNING "net_v1: Failed to change address\n");
    return -EOPNOTSUPP;
  }

  if (map->irq != dev->irq) {
    dev->irq = map->irq;
  }

  return 0;
}

void my_rx(struct net_device* dev, struct my_packet* pkt) {
  struct sk_buff* skb;
  struct my_priv* priv = netdev_priv(dev);

  skb = dev_alloc_skb(pkt->data_size + 2);

  if (!skb) {
    if (printk_ratelimit())
      printk(KERN_NOTICE "net_v1: RX: low on mem - packet dropped\n");

    priv->stats.rx_dropped++;
    goto out;
  }

  skb_reserve(skb, 2);
  memcpy(skb_put(skb, pkt->data_size), pkt->data, pkt->data_size);

  skb->dev = dev;
  skb->protocol = eth_type_trans(skb, dev);
  skb->ip_summed = CHECKSUM_UNNECESSARY;

  priv->stats.rx_packets++;
  priv->stats.rx_bytes += pkt->data_size;
  netif_rx(skb);

out:
  return;
}

static int do_poll(struct napi_struct* napi, int budget) {
  int n_packets = 0;
  struct sk_buff* skb;
  struct my_priv* priv = container_of(napi, struct my_priv, napi);
  struct net_device* dev = priv->dev;
  struct my_packet* pkt;

  while (n_packets < budget && priv->rx_queue) {
    pkt = dequeue_buf(dev);
    skb = dev_alloc_skb(pkt->data_size + 2);

    if (!skb) {
      if (printk_ratelimit())
        printk(KERN_NOTICE "net_v1: Dropped packet\n");

      priv->stats.rx_dropped++;
      release_buffer(pkt);

      continue;
    }

    skb_reserve(skb, 2);
    memcpy(skb_put(skb, pkt->data_size), pkt->data, pkt->data_size);
    skb->dev = dev;
    skb->protocol = eth_type_trans(skb, dev);
    skb->ip_summed = CHECKSUM_UNNECESSARY;
    netif_receive_skb(skb);

    n_packets++;
    priv->stats.rx_packets++;
    priv->stats.rx_bytes += pkt->data_size;
    release_buffer(pkt);
  }

  if (!priv->rx_queue) {
   napi_complete(napi);
   rx_ints(dev, 1);
   return 0;
  }

  return n_packets;
}

static void regular_interrupt(int irq, void* dev_id, struct pt_regs* regs) {
  int s; // Status
  struct my_priv* priv;
  struct my_packet* pkt = NULL;
  struct net_device* dev = (struct net_device*)dev_id;

  if (!dev) return; // ...

  priv = netdev_priv(dev);
  spin_lock(&priv->lock);

  s = priv->status;

  priv->status = 0;

  if (s & MY_RX_INTR) {
    priv->stats.tx_packets++;
    priv->stats.tx_bytes += priv->tx_packet_size;
    dev_kfree_skb(priv->skb);
  }

  spin_unlock(&priv->lock);

  if (pkt) release_buffer(pkt);

  return;
}

static void napi_interrupt(int irq, void* dev_id, struct pt_regs* regs) {
  int s; // Status
  struct my_priv* priv;
  struct net_device* dev= (struct net_device*)dev_id;

  if (!dev) return; // ...

  priv = netdev_priv(dev);
  spin_lock(&priv->lock);

  s = priv->status;
  priv->status = 0;

  if (s & MY_RX_INTR) {
    rx_ints(dev, 0);
    napi_schedule(&priv->napi);
  }

  if (s & MY_TX_INTR) {
    priv->stats.tx_packets++;
    priv->stats.tx_bytes += priv->tx_packet_size;
  }

  spin_unlock(&priv->lock);

  return;
}

static void hw_tx(char* buf, int size, struct net_device* dev) {
  struct iphdr* header; // IPv4 header
  struct net_device* dest; // Destination device
  struct my_priv* priv;
  u32* s_addr, d_addr;
  struct my_packet* tx_buffer;

  if (size < sizeof(struct ethhdr) + sizeof(struct iphdr)) {
    printk("net_v1: %i octest\n", size);
    return;
  }

  if (0) {
    int i;
    printk(KERN_NOTICE "Size is %i\n" KERN_DEBUG "data: ", size);

    for (i=14; i<size; ++i)
      printk(" %02x", buf[i]&0xff);
    printk("\n");
  }

  header = (struct iphdr*)(buf+sizeof(struct ethhdr));
  s_addr = &header->saddr;
  d_addr = &header->daddr;

  ((u8*)s_addr)[2] ^= 1;
  ((u8*)d_addr)[2] ^= 1;

  header->check = 0;
  header->check = ip_fast_csum((unsigned char *)header, header->ihl);

  if (dev == my_devs[0])
    printk(KERN_NOTICE "%08x:%05i --> %08x:%05i\n",
      ntohl(header->saddr), ntohs(((struct tcphdr*)(header+1))->source),
      ntohl(header->daddr), ntohs(((struct tcphdr*)(header+1))->dest));
  else
    printk(KERN_NOTICE "%08X:%05i --> %08x:%05i\n",
      ntohl(header->daddr), ntohs(((struct tcphdr*)(header+1))->dest),
      ntohl(header->saddr), ntohs(((struct tcphdr*)(header+1))->source));

  dest = my_devs[dev == my_devs[0] ? 1 : 0];
  priv = netdev_priv(dest);
  tx_buffer = get_tx_buffer(dev);
  tx_buffer->data_size = size;
  memcpy(tx_buffer->data, buf, size);
  enqueue_buf(dest, tx_buffer);

  if (priv->rx_int_enabled) {
    priv->status |= MY_RX_INTR;
    my_interrupt(0, dest, NULL);
  }

  priv = netdev_priv(dev);
  priv->tx_packet_size = size;
  priv->tx_packet_data = buf;
  priv->status |= MY_TX_INTR;

  if (lockup && ((priv->stats.tx_packets + 1) % lockup) == 0) {
    netif_stop_queue(dev);
    printk(KERN_NOTICE "net_v1: Simulate lockup at %ld, txp %ld\n", jiffies,
      (unsigned long)priv->stats.tx_packets);
  } else
    my_interrupt(0, dev, NULL);
}

int my_tx(struct sk_buff* skb, struct net_device* dev) {
  int size;
  char* data, shortpkt[ETH_ZLEN];
  struct my_priv* priv = netdev_priv(dev);

  data = skb->data;
  size = skb->len;

  if (size < ETH_ZLEN) {
    memset(shortpkt, 0, ETH_ZLEN);
    memcpy(shortpkt, skb->data, skb->len);
    size = ETH_ZLEN;
    data = shortpkt;
  }

  dev->trans_start = jiffies; // Timestamp
  priv->skb = skb;
  hw_tx(data, size, dev);

  return 0;
}

void my_tx_timeout(struct net_device* dev) {
  struct my_priv* priv = netdev_priv(dev);

  printk(KERN_NOTICE "net_v1: Transmit timeout at %ld, latency %ld\n", jiffies,
    jiffies-(dev->trans_start));

  priv->status = MY_TX_INTR;
  my_interrupt(0, dev, NULL);
  priv->stats.tx_errors++;
  netif_wake_queue(dev);

  return;
}

int my_ioctl(struct net_device* dev, struct ifraq* rq, int cmd) {
  printk(KERN_NOTICE "net_v1: ioctl\n");
  return 0;
}

struct net_device_stats* my_stats(struct net_device* dev) {
  struct my_priv* priv = netdev_priv(dev);
  return &priv->stats;
}

int rebuild_header(struct sk_buff* skb) {
  struct ethhdr* header = (struct ethhdr*)skb->data;
  struct net_device* dev = skb->dev;

  memcpy(header->h_source, dev->dev_addr, dev->addr_len);
  memcpy(header->h_dest, dev->dev_addr, dev->addr_len);
  header->h_dest[ETH_ALEN-1] ^= 0x01; // xor 1

  return 0;
}

int my_header(struct sk_buff* skb, struct net_device* dev,
  unsigned short type, const void* daddr, const void* saddr, unsigned size) {
  struct ethhdr* header = (struct ethhdr*)skb_push(skb, ETH_HLEN);

  header->h_proto = htons(type);
  memcpy(header->h_source, saddr ? saddr : dev->dev_addr, dev->addr_len);
  memcpy(header->h_dest, daddr ? daddr : dev->dev_addr, dev->addr_len);
  header->h_dest[ETH_ALEN-1] ^= 0x01; // xor 1

  return (dev->hard_header_len);
}

int my_mtu(struct net_device* dev, int mtu) {
  unsigned long flags;
  struct my_priv* priv = netdev_priv(dev);
  spinlock_t* lock = &priv->lock;

  if ((mtu<68) || (mtu>1500))
    return -EINVAL;

  spin_lock_irqsave(lock, flags);
  dev->mtu = mtu;
  spin_unlock_irqrestore(lock, flags);

  return 0;
}

static const struct net_device_ops my_netdev_ops = {
  .ndo_open = do_open,
  .ndo_stop = do_release,
  .ndo_start_xmit = my_tx,
  .ndo_do_ioctl = my_ioctl,
  .ndo_set_config = my_config,
  .ndo_get_stats = my_stats,
  .ndo_change_mtu = my_mtu,
  .ndo_tx_timeout = my_tx_timeout
};

void do_setup(struct net_device* dev) {
  struct my_priv* priv;
  ether_setup(dev);
  dev->watchdog_timeo = timeout;
  dev->netdev_ops = &my_netdev_ops;
  dev->header_ops = &my_netdev_ops;
  dev->flags |= IFF_NOARP;
  dev->features |= NETIF_F_HW_CSUM;

  priv = netdev_priv(dev);

  if (use_napi)
    netif_napi_add(dev, &priv->napi, do_poll, 2);

  memset(priv, 0, sizeof(struct my_priv));
  spin_lock_init(&priv->lock);
  rx_ints(dev, 1);
  setup_pool(dev);
}
