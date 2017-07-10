#include <linux/ioctl.h>
#include <linux/cdev.h>
#include <linux/semaphore.h>

#undef DEBUGGER
#ifdef DO_DEBUG
#  ifdef __KERNEL__  
/** Kernel space **/
#    define DEBUGGER(fmt, args...) printk(KERN_DEBUG "do_debug: " fmt, ## args)
#  else
/** User space **/
#    define DEBUGGER(fmt, args...) fprintf(stderr, fmt, ## args)
#  endif
#else
#  define DEBUGGER(fmt, args...)
#endif

#undef DEBUGGER
#define DEBUGGER(fmt, args...)

#ifdef DEBUGGER
#  define USE_PROC
#endif
