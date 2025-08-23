#include "TinyEXIF.h"
#include <iostream> // std::cout
#include <fstream>  // std::ifstream
#include <vector>   // std::vector

int main(int argc, const char** argv) {
	if (argc != 2) {
		std::cout << "Usage: TinyEXIF <image_file>" << std::endl;
		return -1;
	}

	// open a stream to read just the necessary parts of the image file
	std::ifstream istream(argv[1], std::ifstream::binary);

	// parse image EXIF and XMP metadata
	TinyEXIF::EXIFInfo imageEXIF(istream);
	if (imageEXIF.Fields)
		std::cout
			<< "Image Description " << imageEXIF.ImageDescription << "\n"
			<< "Image Resolution " << imageEXIF.ImageWidth << "x" << imageEXIF.ImageHeight << " pixels\n"
			<< "Camera Model " << imageEXIF.Make << " - " << imageEXIF.Model << "\n"
			<< "Focal Length " << imageEXIF.FocalLength << " mm\n"
            << "Calibration.FocalLength " << imageEXIF.Calibration.FocalLength << "\n"
            << "Calibration.OpticalCenterX " << imageEXIF.Calibration.OpticalCenterX << std::endl;
	return 0;
}