
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
	int numFailedTests = 0;

	io::CSVReader<2, io::trim_chars<' ', '\t'>, io::no_quote_escape<';'>> in("../../test/data/gas_consumption.csv");

	//in.read_header(io::ignore_extra_column, "date", "consumption_rate");
	const char* dateStr; double consumptionRate;
	while (in.read_row(dateStr, consumptionRate))
	{
		
		std::tm tm = {};
		std::stringstream ss(dateStr);
		ss >> std::get_time(&tm, "%Y-%m-%d");
		auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
		auto date = year_month_day(round<days>(tp));
	}

	return numFailedTests;
}
