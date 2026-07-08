// Native POPGym ConcentrationEasy: one deck matching colors.

#define CONC_NUM_CARDS 52
#define CONC_NUM_VALUES 2
#define CONC_EPISODE_LENGTH 104
#define CONC_CARD_VALUE(i) ((i) % CONC_NUM_VALUES)

#include "../popgym_concentration/popgym_concentration.h"
