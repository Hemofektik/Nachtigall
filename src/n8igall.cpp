
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>


#include "n8igall.h"

namespace n8igall
{
	struct GeoClimateCorrelator : public IGeoClimateCorrelator
	{
		virtual ~GeoClimateCorrelator() {};

	};


	IGeoClimateCorrelator* Create()
	{
		return new GeoClimateCorrelator();
	}
}
