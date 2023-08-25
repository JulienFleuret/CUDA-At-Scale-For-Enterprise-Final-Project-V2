#include <iostream>

#include <npp.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <filesystem>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>


#include "types.h"
#include "utils.h"
#include "magnitude.h"

using namespace std;
using namespace cv;
using namespace cas;






int main(int argc, char* argv[])
{
    String keys = "{help h usage ?|| print this message}"
                  "{input i if|| the input can be: "
                  "                                -the name of a video file."
                  "                                -an image sequence (e.g img_%02d.jpg)."
                  "                                -the url of a video stream (eg. protocol://host:port/script_name?script_params|auth)."
                  "                                -GStreamer pipeline string in gst-launch tool format in case if GStreamer is used as backend Note that each video stream or IP camera feed has its own URL scheme.}"
                  "{output o of|| the output can be:"
                  "                                 -a mp4 file name (e.g. output.mp4)."
                  "                                 -output folder, or output directory hierachy (e.g. output, output/A/B/C)."
                  "                                 -output folder, or output directory hierachy with a file name (e.g. output/output.mp4, output/A/B/C/output.mp4, output/img_%02d.jpg, output/A/B/C/img_%02d.jpg, output/A/B/C/img_%02d.jpg).}"
                  "{batch_size bs|-1| batch size, -1 will provide a default calculation, this may not be suitable for every GPU}";

    CommandLineParser parser(argc, argv, keys);

    parser.about("cudaAtScaleV2.exe --input=[video.mp4, video stream, img_%02d.jpg] --output=[video.mp4, img_%02d.jpg] [--batch_size=positive_number]");

    String input = parser.get<String>("input");
    String output = parser.get<String>("output");
    int batch_size = parser.get<int>("batch_size");

    // To answer the question why does a wrong argument return an EXIT_SUCCESS
    // simply a wrong argument does not affect the normal execution of the program.


    // If the input or the output were not set, the usage message is
    // printed, and the program end properly.
    if(input.empty() || output.empty())
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // If the input does not exists and is not a formating (i.e. img%02d.jpg)
    if(!utils::fs::exists(input) && (input.find("%") == String::npos))
    {
        clog<<"The Specified Input File Does Exists"<<std::endl;
        return EXIT_SUCCESS;
    }

    // If the output is a file, which already exists, the current file
    // is erased, and a new one will be recreated during the initialization
    // of the computation.
    if(utils::fs::exists(output) && !utils::fs::isDirectory(output))
    {
        // Remove the existing file.
        remove(output.c_str());
        // Because the previous command has a latency I do a sleep of 1 second.
        // 1 second is arbitrary, my observations shows that it is more than
        // needed.
        this_thread::sleep_for(1s);
    }
    else if(utils::fs::isDirectory(output))
    {
        // This will contains the filename, which can be a real filename (e.g. smth.avi, that.mp4) or a file formating (e.g. img%02d.png).
        String filename = std::filesystem::path(input).filename().string();

        // The output of the is MP4 file.
        // Why? Simply to avoid having to manage too
        // many fourcc codes.
        if(filename.find(".mp4") == String::npos)
            filename += ".mp4";

        // The filename is then added to the output.
        output = utils::fs::join(output, filename);
    }
    else if (!utils::fs::exists(utils::fs::getParent(output)))
    {
        if(!utils::fs::createDirectories(utils::fs::getParent(output)))
        {
            std::clog<<"Impossible To Create Output Directories or Directory"<<std::endl;
            return EXIT_FAILURE;
        }
    }

    // Create the working object.
    auto obj = compute_magnitude_by_batch_t::create();

    // Initialize the working object.
    obj->init(input, output, batch_size);

    // Run.
    obj->run();


    return EXIT_SUCCESS;
}






