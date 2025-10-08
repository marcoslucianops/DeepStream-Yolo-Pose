#include "interrupt.h"

static guint _CINTR = FALSE;

gboolean
check_for_interrupt(gpointer user_data)
{
  GMainLoop **loop = user_data;
  if (_CINTR) {
    _CINTR = FALSE;
    g_main_loop_quit(*loop);
    return FALSE;
  }
  return TRUE;
}

static void
_intr_handler(int signum)
{
  g_print("DEBUG - KeyboardInterrupt\n");
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_handler = SIG_DFL;
  sigaction(SIGINT, &action, NULL);
  _CINTR = TRUE;
}

void
_intr_setup(void)
{
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_handler = _intr_handler;
  sigaction(SIGINT, &action, NULL);
}
