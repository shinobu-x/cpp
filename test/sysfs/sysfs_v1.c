#include <linux/init.h>
#include <linux/kobject.h>
#include <linux/module.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/slab.h>

MODULE_DESCRIPTION("Sysfs Test Part1");
MODULE_AUTHOR("Shinobu Kinjo");
MODULE_LICENSE("GPL");

struct my_obj {
  struct kobject kobj;
  int a;
  int b;
  int c;
};
#define my_obj_of(x) container_of(x, struct my_obj, kobj)

struct my_attribute {
  struct attribute attr;
  ssize_t (*show)(struct my_obj* me, struct my_attribute* attr, char* buf);
  ssize_t (*store)(struct my_obj* me, struct my_attribute* attr,
    const char* buf, size_t count);
};
#define my_attr_of(x) container_of(x, struct my_attribute, attr)

static ssize_t my_attr_show(struct kobject* kobj, struct attribute* attr,
  char* buf) {

  struct my_attribute* attribute;
  struct my_obj* me;

  printk(KERN_NOTICE "sysfs_v1: my_attr_show");

  attribute = my_attr_of(attr);
  me = my_obj_of(kobj);

  if (!attribute->show)
    return -EIO;

  return attribute->show(me, attribute, buf);
}

static ssize_t my_attr_store(struct kobject* kobj, struct attribute* attr,
  const char* buf, size_t len) {

  struct my_attribute* attribute;
  struct my_obj* me;

  printk(KERN_NOTICE "sysfs_v1: my_attr_store");

  attribute = my_attr_of(attr);
  me = my_obj_of(kobj);

  if (!attribute->store)
    return -EIO;

  return attribute->store(me, attribute, buf, len);
}

static const struct sysfs_ops my_sysfs_ops = {
  .show = my_attr_show,
  .store = my_attr_store,
};

static void my_release(struct kobject* kobj) {

  struct my_obj* me;
  
  printk(KERN_NOTICE "sysfs_v1: my_release");

  me = my_obj_of(kobj);
  kfree(me);
}

static ssize_t my_show(struct my_obj* me, struct my_attribute* attr,
  char* buf) {

  printk(KERN_NOTICE "sysfs_v1: my_show");

  return sprintf(buf, "%d\n", me->a);
}

static ssize_t my_store(struct my_obj* me, struct my_attribute* attr,
  const char* buf, size_t count) {

  printk(KERN_NOTICE "sysfs_v1: my_store");

  sscanf(buf, "%du", &me->a);
  return count;
}

static struct my_attribute a_attr = __ATTR(a, 0666, my_show, my_store);

static ssize_t your_show(struct my_obj* me, struct my_attribute* attr,
  char* buf) {

  int v;

  printk(KERN_NOTICE "sysfs_v1: your_show");

  if (strcmp(attr->attr.name, "b") == 0)
    v = me->b;
  else
    v = me->c;

  return sprintf(buf, "%d\n", v);
}

static ssize_t your_store(struct my_obj* me, struct my_attribute* attr,
  const char* buf, size_t count) {

  int v;

  printk(KERN_NOTICE "sysfs_v1: your_store");

  sscanf(buf, "%d\n", &v);

  if (strcmp(attr->attr.name, "b") == 0)
    me->b = v;
  else
    me->c = v;

  return count;
}

static struct my_attribute b_attr = __ATTR(b, 0666, your_show, your_store);
static struct my_attribute c_attr = __ATTR(c, 0666, your_show, your_store);

static struct attribute* my_default_attr[] = {
  &a_attr.attr,
  &b_attr.attr,
  &c_attr.attr,
  NULL,
};

static struct kobj_type my_type = {
  .sysfs_ops = &my_sysfs_ops,
  .release = my_release,
  .default_attrs = my_default_attr,
};

static struct kset* my_kset;
static struct my_obj *my_a;
static struct my_obj *my_b;
static struct my_obj *my_c;

static struct my_obj* do_create(const char* name) {

  struct my_obj* me;
  int r;

  printk(KERN_NOTICE "sysfs_v1: do_create");

  me = kzalloc(sizeof(*me), GFP_KERNEL);

  if (!me)
    return NULL;

  r = kobject_init_and_add(&me->kobj, &my_type, NULL, "%s", name);

  if (r) {
    kobject_put(&me->kobj);
    return NULL;
  }

  kobject_uevent(&me->kobj, KOBJ_ADD);

  return me;
}

static void do_destroy(struct my_obj* me) {

  printk(KERN_NOTICE "sysfs_v1: do_destroy");

  kobject_put(&me->kobj);
}

static int __init my_init(void) {

  printk(KERN_NOTICE "sysfs_v1: do_init");

  my_kset = kset_create_and_add("my_kset", NULL, kernel_kobj);

  if (!my_kset)
    return -ENOMEM;

  my_a = do_create("a");
  if (!my_a)
    goto a_error;

  my_b = do_create("b");
  if (!my_b)
    goto b_error;

  my_c = do_create("c");
  if (!my_c)
    goto c_error;

  return 0;

b_error:
  do_destroy(my_b);
c_error:
  do_destroy(my_c);
a_error:
  do_destroy(my_a);
}

static void __exit my_exit(void) {

  printk(KERN_NOTICE "sysfs_v1: do_exit");

  do_destroy(my_b);
  do_destroy(my_c);
  do_destroy(my_a);
  kset_unregister(my_kset);
}

module_init(my_init);
module_exit(my_exit);
