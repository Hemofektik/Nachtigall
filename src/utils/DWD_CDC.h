#pragma once

#include "../n8igallcore.h"

namespace n8igall
{
	// Deutsche Wetter Dienst - Climate Data Center
	// facilitates access to historic data archive of DWD
	class DWD_CDC
	{
	public:
		DWD_CDC();
		DWD_CDC(const DWD_CDC&) = delete;
		~DWD_CDC();
	};
}
