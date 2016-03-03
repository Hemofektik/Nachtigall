
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>


#include "n8igall.h"

namespace n8igall
{
	struct Brain : public IBrain
	{
		virtual ~Brain() {};

	};


	IBrain* Create()
	{
		return new Brain();
	}
}