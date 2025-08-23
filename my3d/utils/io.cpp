#include "utils/io.h"

namespace my3d {
namespace util {

std::vector<std::string> collectFiles(const std::string &directory, 
                                      const std::string &extension) {
    std::vector<std::string> files; 
    boost::filesystem::path dir(directory); 
    for (const auto& iter : boost::filesystem::directory_iterator(dir)) {
        if (boost::filesystem::is_directory(iter.path())) {
            continue;
        }
        if (iter.path().extension() == extension) {
            files.push_back(iter.path().string()); 
        }
    }

    return files; 
}

std::vector<std::string> collectFiles(const std::string &directory, 
                                      const std::unordered_set<std::string> &extensions) {
    std::vector<std::string> files; 
    boost::filesystem::path dir(directory); 
    for (const auto& iter : boost::filesystem::directory_iterator(dir)) {
        if (boost::filesystem::is_directory(iter.path())) {
            continue;
        }
        if (extensions.count(iter.path().extension().string()) != 0) {
            files.push_back(iter.path().string()); 
        }
    }

    return files; 
}

bool readStringFromFStream(std::string &str, 
                           std::ifstream &ifs) {
    str.clear(); 
    if (!ifs.is_open() || !ifs.good()) {
        return false; 
    } 

    char ch; 
    ifs.read(&ch, sizeof(char)); 
    while (ch != '\0' && ifs.good()) {
        str.push_back(ch); 
        ifs.read(&ch, sizeof(char)); 
    } 

    return ifs.good(); 
}

}
}
