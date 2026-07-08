// Native POPGym ConcentrationMedium: two decks matching colors.

#define CONC_NUM_CARDS 104
#define CONC_NUM_VALUES 2
#define CONC_EPISODE_LENGTH 208
#define CONC_CARD_VALUE(i) ((i) % CONC_NUM_VALUES)

#include "../popgym_concentration/popgym_concentration.h"
