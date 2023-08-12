#include <iostream>
#include <cstdlib>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/quality.hpp>


using namespace cv;

int main()
{

	try{
		
		String pwd = cv::utils::fs::getcwd();

		String execute_bin = format("%s/../bin/cudaAtScaleV2.exe --ifd=%s/../data --if=squirrel_cls.jpg --ofd=%s/../output --of=tgt.jpg --nd=1", pwd.c_str(), pwd.c_str(), pwd.c_str());
		String ref_filename = format("%s/../data/unit_ref.jpg", pwd.c_str());
		String tgt_filename = format("%s/../output/tgt.jpg", pwd.c_str());
		
		std::cout<<"Target filenames: "<<ref_filename<<std::endl;
		
		std::cout<<"Does Reference Filename Exists: "<<utils::fs::exists(ref_filename)<<std::endl;

		int sys_ret = std::system(execute_bin.c_str());
		
		if(sys_ret)
		   std::clog<<"An Error Has Occured During The Call Of The Function cudaAtScaleV2.exe"<<std::endl;
		
		
		Mat ref = imread(ref_filename, IMREAD_UNCHANGED);
		Mat tgt = imread(tgt_filename, IMREAD_UNCHANGED);

		if(ref.empty())
		{
			std::clog<<"Wrong Reference Path"<<std::endl;
		}

		if(tgt.empty())
		{
			std::clog<<"Wrong Target Path"<<std::endl;
		}

		if(ref.empty() || tgt.empty())
			return EXIT_SUCCESS;

		Ptr<quality::QualityMSE> metric = quality::QualityMSE::create(ref);

		Scalar delta = metric->compute(tgt);
	
		int ret = (delta(0) <= std::numeric_limits<float>::epsilon()) ? EXIT_SUCCESS : EXIT_FAILURE;
	
		if(ret == EXIT_SUCCESS)
		   std::cout<<"SUCCESS!";
		else
		   std::cout<<"FAILURE!";
		std::cout<<std::endl;
	
		return ret;
	
	}
	catch(std::exception& err)
	{
		std::clog<<"An Unexpected Error Has Occured: "<<err.what()<<std::endl;
	}
	
	return EXIT_FAILURE;
}
