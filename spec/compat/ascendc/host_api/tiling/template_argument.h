// Version-skew shim.
// ops-transformer expects <ascendc/host_api/tiling/template_argument.h>,
// but CANN 9.1.0-beta.3 ships this header at <tiling/template_argument.h>.
#pragma once
#include "tiling/template_argument.h"
