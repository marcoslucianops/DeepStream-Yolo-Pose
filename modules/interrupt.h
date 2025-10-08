#ifndef __INTERRUPT_H__
#define __INTERRUPT_H__

#include <glib.h>

gboolean check_for_interrupt(gpointer user_data);

void _intr_setup(void);

#endif
