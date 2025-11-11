
#include "MnistReader.h"

MnistReader::MnistReader(std::string train_images, std::string train_labels, std::string test_images,  std::string test_labels)
:
train_images_(std::move(train_images)),
train_labels_(std::move(train_labels)),
test_images_(std::move(test_images)),
test_labels_(std::move(test_labels))
{
}

void MnistReader::load()
{
    auto trX = load_images(train_images_);
    auto trY = load_labels(train_labels_);
    auto teX = load_images(test_images_);
    auto teY = load_labels(test_labels_);

    if (trX.size() != trY.size())
    {
        throw std::runtime_error("train images/labels mismatch");
    }
    if (teX.size() != teY.size())
    {
        throw std::runtime_error("test images/labels mismatch");
    }

    training_.clear(); test_.clear();
    training_.reserve(trX.size());
    for (size_t i = 0; i < trX.size(); ++i)
    {
        training_.emplace_back(std::move(trX[i]), std_basis_vec(trY[i], 10));
    }

    test_.reserve(teX.size());
    for (size_t i = 0; i < teX.size(); ++i)
    {
        test_.emplace_back(std::move(teX[i]), std_basis_vec(teY[i], 10));
    }
}

std::vector<VectorXd> MnistReader::load_images(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open images failed: " + path);
    if (be_u32(f) != 2051) throw std::runtime_error("bad images magic");
    uint32_t num = be_u32(f), rows = be_u32(f), cols = be_u32(f);
    uint32_t dim = rows*cols;

    std::vector<VectorXd> out; out.reserve(num);
    std::vector<uint8_t> buf(dim);
    for (uint32_t i=0;i<num;++i) {
        f.read(reinterpret_cast<char*>(buf.data()), dim);
        VectorXd v(dim);
        for (uint32_t j=0;j<dim;++j) v(j) = double(buf[j]) / 255.0;
        out.push_back(std::move(v));
    }
    return out;
}


std::vector<int> MnistReader::load_labels(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open labels failed: " + path);
    if (be_u32(f) != 2049) throw std::runtime_error("bad labels magic");
    uint32_t num = be_u32(f);
    std::vector<int> out(num);
    for (uint32_t i=0;i<num;++i) { unsigned char c; f.read(reinterpret_cast<char*>(&c),1); out[i]=int(c); }
    return out;
}


void MnistReader::print(const VectorXd& img, int label, int rows, int cols)
{
    static const char* shades = " .:-=+*#%@";   // 10 gradations of intensity

    std::cout << "Label: " << label << "\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double v = img(r * cols + c);       // value in [0,1]
            int idx = std::clamp(int(v * 9.999), 0, 9);
            std::cout << shades[idx];
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

