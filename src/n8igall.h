#pragma once

#include "n8igallcore.h"

namespace n8igall
{
	struct IBrain
	{
		virtual ~IBrain() {};

		static IBrain* Create();
	};
}
