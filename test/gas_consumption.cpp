
#include "../src/n8igall.h"
#include <csv.h>
#include <date.h>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace n8igall;

using namespace date;
using namespace std::chrono;

int main(int argc, const char* argv[])
{
	unique_ptr<IGeoTimeSeries> gts(IGeoTimeSeries::Create(13.2759793, 52.6673332));

	// read the data from csv file
	{
		io::CSVReader<2, io::trim_chars<' ', '\t'>, io::no_quote_escape<';'>> in("../../test/data/gas_consumption.csv");
		const char* dateStr; double consumptionRate;
		while (in.read_row(dateStr, consumptionRate))
		{
			std::tm tm = {};
			std::stringstream ss(dateStr);
			ss >> std::get_time(&tm, "%Y-%m-%d");
			const auto timePoint = std::chrono::system_clock::from_time_t(std::mktime(&tm));
			const auto dayPoint = round<days>(timePoint);
			//auto date = year_month_day();

			auto sample = ISample::Create();
			sample->SetValue(0, consumptionRate);
			gts->AddTimeSample(dayPoint, sample);
		}
	}

	// TODO: pass time series to IGeoClimateCorrelator and see what happens next

	return 0;
}
