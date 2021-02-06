#include <iostream>
#include <string_view>

auto main(int argc, char* argv[]) -> int {
    if (argc < 2) {
        std::cerr << "Usage: mlc path_to_data\n";
        return 1;
    }
    auto path = std::string_view{argv[1]};  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::clog << "path: " << path << "\n";
    return 0;
}
