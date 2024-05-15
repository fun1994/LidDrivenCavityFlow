#include "LidDrivenCavityFLow.h"

void test(std::string grid, std::string Re) {
	LidDrivenCavityFLow LDCF(grid, std::stod(Re), 64, 64, 1e-3, 1e-3, 1e-3, 1e-8);
	Data data;
	LDCF.solve(data);
	data.save(grid + "/Re=" + Re);
}

int main() {
	test("staggered", "100");
	test("staggered", "400");
	test("staggered", "1000");
	test("collocated", "100");
	test("collocated", "400");
	test("collocated", "1000");
	return 0;
}
