// Native POPGym CountRecallHard: four decks counting ranks.

#define CR_NUM_VALUES 13
#define CR_DECK_SIZE 208
#define CR_CARD_VALUE(i) (((i) % 52) / 4)

#include "../popgym_count_recall/popgym_count_recall.h"
