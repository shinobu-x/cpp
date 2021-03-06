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

static void do_tx_timeout(struct net_device* dev);
static void (*irq_interrupt)(int, void*, struct pt_regs*);

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
    pkt->next = priv->packet_pool;
    priv->packet_pool = pkt;
  }
}

void do_teardown_pool(struct net_device* dev) {
  struct priv_t* priv = netdev_priv(dev);
  struct packet_t* pkt;

  while ((pkt = priv->packet_pool)) {
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

static void rx_ints(struct net_device* dev, int enable) {
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

int set_config(struct net_device* dev, struct ifmap* map) {
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
  u32* saddr;
  u32* daddr;
  struct packet_t* tx_buf;

  if (len < sizeof(struct ethhdr) + sizeof(struct iphdr)) {
    printk("net_v2: %i octet\n", len);
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

  if (dev == net_devs[0])
    printk(KERN_NOTICE "%08x:%05i --> %08x:%05i\n",
      ntohl(header->saddr), ntohs(((struct tcphdr*)(header+1))->source),
      ntohl(header->daddr), ntohs(((struct tcphdr*)(header+1))->dest));
  else
    printk(KERN_NOTICE "%08x:%05i --> %08x:%05i\n",
      ntohl(header->daddr), ntohs(((struct tcphdr*)(header+1))->dest),
      ntohl(header->saddr), ntohs(((struct tcphdr*)(header+1))->source));

  dest = net_devs[dev == net_devs[0] ? 1 : 0];
  priv = netdev_priv(dest);
  tx_buf = get_tx_buffer(dev);
  tx_buf->data_len = len;
  memcpy(tx_buf->data, buf, len);
  enqueue_buffer(dest, tx_buf);

  if (priv->rx_int_enabled) {
    priv->status |= RX_INTR;
    irq_interrupt(0, dest, NULL);
  }

  if (lockup && ((priv->stats.tx_packets + 1) % lockup) == 0) {
    netif_stop_queue(dev);
    printk(KERN_NOTICE "net_v2: Simulate lockup at %ld, txp %ld\n", jiffies,
      (unsigned long)priv->stats.tx_packets);
  } else
    irq_interrupt(0, dev, NULL);
}

int do_tx(struct sk_buff* skb, struct net_device* dev) {
  int len;
  char* data, short_packet[ETH_ZLEN];
  struct priv_t* priv = netdev_priv(dev);

  data = skb->data;
  len = skb->len;

  if (len < ETH_ZLEN) {
    memset(short_packet, 0, ETH_ZLEN);
    memcpy(short_packet, skb->data, skb->len);
    len = ETH_ZLEN;
    data = short_packet;
  }

  dev->trans_start = jiffies;  // Timestamp
  priv->skb = skb;
  hw_tx(data, len, dev);

  return 0;
}

void do_tx_timeout(struct net_device* dev) {
  struct priv_t* priv = netdev_priv(dev);

  printk(KERN_NOTICE "net_v2: Transmit timeout at %ld, latency %ld\n", jiffies,
    jiffies-(dev->trans_start));

  priv->status = TX_INTR;
  irq_interrupt(0, dev, NULL);
  priv->stats.tx_errors++;
  netif_wake_queue(dev);  // Transmit packets again

  return;
}

int do_ioctl(struct net_device* dev, struct ifraq* rq, int cmd) {
  printk(KERN_NOTICE "net_v2: ioctl\n");
  return 0;
}

struct net_device_stats* get_stats(struct net_device* dev) {
  struct priv_t* priv = netdev_priv(dev);
  return &priv->stats;
}

int rebuild_header(struct sk_buff* skb) {
  struct ethhdr* header = (struct ethhdr*)skb->data;
  struct net_device* dev = skb->dev;

  memcpy(header->h_source, dev->dev_addr, dev->addr_len);
  memcpy(header->h_dest, dev->dev_addr, dev->addr_len);
  header->h_dest[ETH_ALEN-1] ^= 0x01;  // XOR 1

  return (dev->hard_header_len);
}

int create_header(struct sk_buff* skb, struct net_device* dev,
  unsigned short type, const void* daddr, const void* saddr, unsigned len) {
  // skb_push: Add data to beginning of the packet
  struct ethhdr* header = (struct ethhdr*)skb_push(skb, ETH_HLEN);

  header->h_proto = htons(type);
  memcpy(header->h_source, saddr ? saddr : dev->dev_addr, dev->addr_len);
  memcpy(header->h_dest, daddr ? daddr : dev->dev_addr, dev->addr_len);
  header->h_dest[ETH_ALEN-1] ^= 0x01;  // XOR

  return (dev->hard_header_len);
}

int set_mtu(struct net_device* dev, int mtu) {
  unsigned long flags;
  struct priv_t* priv = netdev_priv(dev);
  spinlock_t* l = &priv->l;

  if ((mtu<68) || (mtu>1500))
    return -EINVAL;

  spin_lock_irqsave(l, flags);
  dev->mtu = mtu;
  spin_unlock_irqrestore(l, flags);

  return 0;
}

static const struct header_ops header_ops_t = {
  .create = create_header,
  .rebuild = rebuild_header
};

static const struct net_device_ops device_ops_t = {
  .ndo_open = do_open,
  .ndo_stop = do_release,
  .ndo_start_xmit = do_tx,
  .ndo_do_ioctl = do_ioctl,
  .ndo_set_config = set_config,
  .ndo_get_stats = get_stats,
  .ndo_change_mtu = set_mtu,
  .ndo_tx_timeout = do_tx_timeout
};

void do_setup(struct net_device* dev) {
  struct priv_t* priv;
  dev->watchdog_timeo = timeout;
  dev->netdev_ops = &device_ops_t;
  dev->header_ops = &header_ops_t;
  dev->flags |= IFF_NOARP;
  dev->features |= NETIF_F_HW_CSUM;

  priv = netdev_priv(dev);

  if (use_napi)
    netif_napi_add(dev, &priv->napi, do_poll, 2);

  memset(priv, 0, sizeof(struct priv_t));
  spin_lock_init(&priv->l);
  rx_ints(dev, 1);
  setup_pool(dev);
}

void do_cleanup(void) {
  int i;

  for (i=0; i<2; ++i) {
    if (net_devs[i]) {
      unregister_netdev(net_devs[i]);
      do_teardown_pool(net_devs[i]);
      free_netdev(net_devs[i]);
    }
  }
  return;
}

int do_init(void) {
  int result, i, r = -ENOMEM;

  irq_interrupt = use_napi ? napi_interrupt : regular_interrupt;

  net_devs[0] = alloc_netdev(sizeof(struct priv_t), "TEST_IF%d", do_setup);
  net_devs[1] = alloc_netdev(sizeof(struct priv_t), "TEST_IF%d", do_setup);

  if (net_devs[0] == NULL || net_devs[1] == NULL)
    goto out;

  r = -ENODEV;

  for (i=0; i<2; ++i)
    if ((result = register_netdev(net_devs[i])))
      printk(KERN_NOTICE "net_v2: Failed to register device [%i] \"%s\"\n",
        result, net_devs[i]->name);
    else
      r = 0;

out:
  if (r)
    do_cleanup();
  return r;
}

// ******

module_init(do_init);
module_exit(do_cleanup);
