#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <memory>
#include <date.h>

namespace n8igall
{
	typedef char			char_t;

	typedef std::string		string; // used to store UTF-8 strings

	typedef int8_t			s8;
	typedef int16_t			s16;
	typedef int32_t			s32;
	typedef int64_t			s64;
	typedef uint8_t			u8;
	typedef uint16_t		u16;
	typedef uint32_t		u32;
	typedef uint64_t		u64;

	typedef std::size_t		size;

	typedef float			f32;
	typedef double			f64;

	typedef date::day_point dayPoint;
}
