#include <ag/ag.hpp>
#include <cassert>
#include <cmath>
#include <gsl/gsl>
#include <iostream>
#include <mnist/mnist_reader.hpp>
#include <optional>
#include <string_view>

using namespace ag;

namespace {

struct RunningMean {
    RunningMean(Float retain_rate) : rr(retain_rate) {}

    auto update(Float val) { value = !value ? val : (*value * rr + val * (1 - rr)); }

    auto get() {
        Expects(value);
        return *value;
    }

private:
    std::optional<Float> value{};
    Float rr{};
};

}  // namespace

auto main(int argc, char* argv[]) -> int {
    if (argc < 2) {
        std::cerr << "Usage: mlc path_to_data\n";
        return 1;
    }
    auto path = std::string_view{argv[1]};  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    constexpr auto image_size = 28 * 28;
    constexpr auto batch_size = 1;
    constexpr auto lr = 1e-3F;

    auto x = Matrix<batch_size, image_size>{};
    auto y = VectorCol<batch_size>{};
    auto weights = VectorCol<image_size>{VectorCol<image_size>::ValueT::ones()};
    auto bias = VectorCol<batch_size>{VectorCol<batch_size>::ValueT::zeros()};

    auto y_pred = matmul(x, weights) + bias;
    auto loss = avg((y - y_pred) * (y - y_pred));

    auto dataset = mnist::read_dataset<std::vector, std::vector, std::uint8_t, std::uint8_t>(path.data());
    auto& images = dataset.training_images;
    auto& labels = dataset.training_labels;
    Expects(images.size() == labels.size());
    Expects(images.size() % batch_size == 0);
    Expects(images[0].size() == image_size);

    auto convertImagesBatchTo = [](auto& value, auto& it) {
        for (auto row = 0; row < batch_size; ++row, ++it) {
            for (auto col = 0; col < it->size(); ++col) { value.item(row, col) = 1.F * (*it)[col]; }
        }
    };

    auto convertLabelsBatchTo = [](auto& value, auto& it) {
        for (auto row = 0; row < batch_size; ++row, ++it) { value.item(row) = 1.F * (*it); }
    };

    std::cout << "Start calculating" << std::endl;
    constexpr auto max_epoch = 5;
    for (auto epoch = 0; epoch < max_epoch; ++epoch) {
        auto mean_loss = RunningMean{.99F};
        auto iti = std::begin(images);
        auto itl = std::begin(labels);
        while (iti != std::end(images) && itl != std::end(labels)) {
            convertImagesBatchTo(x.value(), iti);
            convertLabelsBatchTo(y.value(), itl);
            mean_loss.update(loss.value(true).item());
            if (!std::isfinite(mean_loss.get())) {
                std::cerr << "Mean Loss is not finite at epoch " << epoch << " and batch nr "
                          << std::distance(std::begin(images), iti) << std::endl;
                return 2;
            }
            loss.backprop();
            weights.value() = weights.value() - (weights.grad() * lr);
            bias.value() = bias.value() - (bias.grad() * lr);
            loss.zerograd();
        }
        std::cout << "Loss after epoch " << epoch << ": " << mean_loss.get() << std::endl;
    }

    return 0;
}
