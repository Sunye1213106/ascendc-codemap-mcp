/**
 * Compatibility hole-fill for CANN 9.1.0-beta.3 unpack.
 * ``acl/acl_base.h`` includes this header, but the extracted npu-runtime
 * package does not ship the file. Runtime types live in ``acl_base_rt.h``.
 */
#ifndef INC_EXTERNAL_ACL_ACL_BASE_MDL_H_
#define INC_EXTERNAL_ACL_ACL_BASE_MDL_H_

#include "acl/acl_base_rt.h"

#endif
