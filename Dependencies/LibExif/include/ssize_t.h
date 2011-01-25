#pragma once

#ifdef _WIN64
	typedef __int64 ssize_t;
#else
	typedef _w64 int ssize_t;
#endif