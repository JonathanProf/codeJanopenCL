#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include "functions.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 200

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define DEBUG_CASE 0

char* readKernel( const char *filename ){
		
		FILE *fp;
		char *fileData;
		long fileSize;
		
		/* Open the file */
		fp = fopen(filename, "r");
		if( !fp ){
			printf( "Could not open file: %s\n", filename );
			exit(-1);
		}
		
		/* Determine the file size */
		if( fseek(fp, 0, SEEK_END)){
			printf("Error reading the file\n");
			exit(-1);
		}
		fileSize = ftell(fp);
		if(fileSize < 0){
			printf("Error reading the file\n");
			exit(-1);
		}
		if( fseek(fp, 0, SEEK_SET)){
			printf("Error reading the file\n");
			exit(-1);
		}
		
		/* Read the contents */
		fileData = (char*)malloc(fileSize+1);
		if( !fileData ){
			exit(-1);
		}
		if( fread(fileData, fileSize, 1, fp) != 1 ){
			printf("Error reading the file\n");
			exit(-1);
		}
		
		/* Terminate the string */
		fileData[fileSize] = '\0';
		
		/* Close the file */
		if( fclose(fp) ){
			printf("Error closing the file\n");
			exit(-1);
		}
		return fileData;
}

#define CASE_CL_ERROR(NAME) case NAME: return #NAME;

const char* opencl_error_to_str (cl_int error) {
	switch(error) {
		CASE_CL_ERROR(CL_SUCCESS)
		CASE_CL_ERROR(CL_DEVICE_NOT_FOUND)
		CASE_CL_ERROR(CL_DEVICE_NOT_AVAILABLE)
		CASE_CL_ERROR(CL_INVALID_PROGRAM)
		CASE_CL_ERROR(CL_INVALID_CONTEXT)
		CASE_CL_ERROR(CL_INVALID_VALUE)
		CASE_CL_ERROR(CL_INVALID_QUEUE_PROPERTIES)
		CASE_CL_ERROR(CL_OUT_OF_RESOURCES)
		CASE_CL_ERROR(CL_INVALID_DEVICE)
		CASE_CL_ERROR(CL_INVALID_BINARY)
		CASE_CL_ERROR(CL_INVALID_BUILD_OPTIONS)
		CASE_CL_ERROR(CL_INVALID_OPERATION)
		CASE_CL_ERROR(CL_COMPILER_NOT_AVAILABLE)
		CASE_CL_ERROR(CL_BUILD_PROGRAM_FAILURE)
		CASE_CL_ERROR(CL_OUT_OF_HOST_MEMORY)
		CASE_CL_ERROR(CL_INVALID_PROPERTY)
	default:
	return "UNKNOWN ERROR CODE";
	}
}

#define CHECK_STATUS(status) \
			if (status != CL_SUCCESS) {\
				fprintf(stderr,\
				"OpenCL error in file %s line %d, error code %s\n",\
				__FILE__,\
				__LINE__,\
				opencl_error_to_str(status));\
				exit(0);\
			}


using namespace std;

#define PATH_SAMPLES_POISSON "./inputSamples/%05d_inputSpikesPoisson.csv"
#define PATH_PARAMETERS_NET "./BD400/"
#define PATH_RESULTS_NET "./classification/"

int main()
{
	
	const char *mainKernel = readKernel("mainKernel.cl");
	
	
	const int NUM_NEURONS = 400;
	const int numPixels = 784;
	int indexWin = NUM_NEURONS;
    //! =====     =====     =====
    //! Variables Inicialization
    //! =====     =====     =====
    const float single_sample_time =   250; // [ms]
    int timeS[250] = {0};

    float *theta;
    float *weightsXeAe; // connections Xe -> Ae
    float *weightsAeAi; // connections Ae -> Ai
    float *weightsAiAe; // connections Ae <- Ai
    bool  *input_sample;

    // Check for spiking neurons
    bool *spikesXePre; // Spike occurrences Input
    bool *spikesXePos; // Spike occurrences Input
    bool spikes_Ae_Ai_pre[NUM_NEURONS] = {0};
    bool spikes_Ae_Ai_pos[NUM_NEURONS] = {0};
    bool spikes_Ai_Ae_pre[NUM_NEURONS] = {0};
    bool spikes_Ai_Ae_pos[NUM_NEURONS] = {0};


    unsigned short int *spike_count; // for xxxx neurons
    unsigned short int *assignments; // for xxxx neurons
    float *proportions; // for xxxx neurons (rows) and 10 digits (columns)

    unsigned short int digits[10] = {0};

    float *vE;
    float *vI;

    int *refrac_countE;       // Refractory period counters
    int *refrac_countI;       // Refractory period counters
    //int dt = 1;

    //! Constants definition
    
    float v_rest_e = -65.0;     // [mV]
    //float v_reset_e = -60.0;    // [mV]
    //float v_thresh_e = -52.0;   // [mV]
    //int refrac_e = 5;           // [ms] Refractory time

    float v_rest_i = -60.0;     // [mV]
    //float v_reset_i = -45.0;    // [mV]
    //float v_thresh_i = -40.0;   // [mV]
    //int refrac_i = 2;           // [ms] Refractory time
    

    //! [Step 1] Data structure initialization
    for (int indx = 0; indx < single_sample_time; ++indx) {
        timeS[indx] = indx;
    }

    spikesXePre = new(std::nothrow) bool[784]{0};
    assert(spikesXePre != nullptr);

    spikesXePos = new(std::nothrow) bool[784]{0};
    assert(spikesXePos != nullptr);

    vE = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vE != nullptr );

    vI = new(std::nothrow) float[NUM_NEURONS]{0.0};
    assert( vI != nullptr );

    for (int indx = 0; indx < NUM_NEURONS; ++indx) {
        vE[indx] = v_rest_e;
        vI[indx] = v_rest_i;
    }

    refrac_countE = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countE != nullptr );

    refrac_countI = new(std::nothrow) int[NUM_NEURONS]{0};
    assert( refrac_countI != nullptr );

    input_sample = new(std::nothrow) bool[250*784];
    assert( input_sample != nullptr );

    spike_count = new(std::nothrow) unsigned short int[NUM_NEURONS]{0};
    assert( spike_count != nullptr );

    //! [Step 2] Loading data from files

    weightsXeAe = new(std::nothrow) float[784*NUM_NEURONS]{0};
    assert( weightsXeAe != nullptr );

    std::string filename = std::string(PATH_PARAMETERS_NET) + "XeAe.csv";
    getWeights(weightsXeAe, filename, 784, NUM_NEURONS);

    weightsAeAi = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAeAi != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "AeAi.csv";
    getWeights(weightsAeAi, filename, NUM_NEURONS, NUM_NEURONS);

    weightsAiAe = new(std::nothrow) float[NUM_NEURONS*NUM_NEURONS]{0};
    assert( weightsAiAe != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "AiAe.csv";
    getWeights(weightsAiAe, filename, NUM_NEURONS, NUM_NEURONS);

    //! theta values are loaded into the group of excitatory neurons
    theta = new(std::nothrow) float[NUM_NEURONS]{0};
    assert( theta != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "theta.csv";
    getTheta( theta, filename );

    assignments = new(std::nothrow) unsigned short int[NUM_NEURONS]{0};
    assert( assignments != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "assignments.csv";
    getAssignments( assignments, filename );

    proportions = new(std::nothrow) float[NUM_NEURONS*10]{0};
    assert( proportions != nullptr );

    filename = std::string(PATH_PARAMETERS_NET) + "proportions.csv";
    getProportions( proportions, NUM_NEURONS, 10, filename );
 
    // Get platform and device information
	cl_platform_id * platforms = NULL;
	cl_uint     num_platforms;
	
	//Set up the Platform
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_STATUS(clStatus);
	
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_STATUS(clStatus);

	//Get the devices list and choose the device you want to run on
	cl_device_id     *device_list = NULL;
	cl_uint           num_devices;

	clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
	CHECK_STATUS(clStatus);
	device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
	CHECK_STATUS(clStatus);

	// Create one OpenCL context for each device in the platform
	cl_context context;
	context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);
	CHECK_STATUS(clStatus);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_list[0], 0, &clStatus);
	CHECK_STATUS(clStatus);

	cl_program program = clCreateProgramWithSource(context, 1,(const char **)&mainKernel, NULL, &clStatus);
	CHECK_STATUS(clStatus);

	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	CHECK_STATUS(clStatus);

	cl_kernel kernel = clCreateKernel(program, "mainKernel", &clStatus);
	CHECK_STATUS(clStatus);

	cl_mem cl_mem_theta                  	= clCreateBuffer(context, CL_MEM_READ_ONLY, (NUM_NEURONS) * sizeof(float), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_weightsXeAe       	= clCreateBuffer(context, CL_MEM_READ_ONLY, (numPixels*NUM_NEURONS) * sizeof(float), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_weightsAeAi        	= clCreateBuffer(context, CL_MEM_READ_ONLY, (NUM_NEURONS*NUM_NEURONS) * sizeof(float), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_weightsAiAe        	= clCreateBuffer(context, CL_MEM_READ_ONLY, (NUM_NEURONS*NUM_NEURONS) * sizeof(float), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_input_sample      	= clCreateBuffer(context, CL_MEM_READ_ONLY, (single_sample_time*numPixels) * sizeof(bool), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_spikesXePre 			= clCreateBuffer(context, CL_MEM_READ_WRITE, numPixels * sizeof(bool), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_spikesXePos        	= clCreateBuffer(context, CL_MEM_READ_WRITE, (numPixels) * sizeof(bool), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_spikes_Ae_Ai_pre = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_NEURONS * sizeof(bool), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_spikes_Ae_Ai_pos = clCreateBuffer(context, CL_MEM_READ_WRITE, (NUM_NEURONS) * sizeof(bool), NULL, &clStatus);
	CHECK_STATUS(clStatus);	
	cl_mem cl_mem_spikes_Ai_Ae_pre = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_NEURONS * sizeof(bool), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_spikes_Ai_Ae_pos = clCreateBuffer(context, CL_MEM_READ_WRITE, (NUM_NEURONS) * sizeof(bool), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_spike_count			= clCreateBuffer(context, CL_MEM_READ_WRITE, (NUM_NEURONS) * sizeof(unsigned short int), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_assignments			= clCreateBuffer(context, CL_MEM_READ_ONLY, (NUM_NEURONS) * sizeof(unsigned short int), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_digits					= clCreateBuffer(context, CL_MEM_READ_WRITE, (10) * sizeof(unsigned short int), NULL, &clStatus);
	CHECK_STATUS(clStatus);		
	cl_mem cl_mem_vE = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_NEURONS * sizeof(float), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_vI = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_NEURONS * sizeof(float), NULL, &clStatus);
	CHECK_STATUS(clStatus);
	cl_mem cl_mem_refrac_countE = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_NEURONS * sizeof(int), NULL, &clStatus);
	CHECK_STATUS(clStatus);	
	cl_mem cl_mem_refrac_countI = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_NEURONS * sizeof(int), NULL, &clStatus);
	CHECK_STATUS(clStatus);

	
	clStatus = clSetKernelArg(kernel, 0, sizeof(int), (void *)&NUM_NEURONS);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 1, sizeof(int), (void *)&timeS[0]);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 2, sizeof(int), (void *)&indexWin);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&cl_mem_theta);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&cl_mem_weightsXeAe);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&cl_mem_weightsAeAi);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&cl_mem_weightsAiAe);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&cl_mem_input_sample);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&cl_mem_spikesXePre);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&cl_mem_spikesXePos);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&cl_mem_spikes_Ae_Ai_pre);
	CHECK_STATUS(clStatus);		
	clStatus = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *)&cl_mem_spikes_Ae_Ai_pos);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *)&cl_mem_spikes_Ai_Ae_pre);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *)&cl_mem_spikes_Ai_Ae_pos);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 14, sizeof(cl_mem), (void *)&cl_mem_spike_count);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 15, sizeof(cl_mem), (void *)&cl_mem_assignments);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 16, sizeof(cl_mem), (void *)&cl_mem_digits);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 17, sizeof(cl_mem), (void *)&cl_mem_vE);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 18, sizeof(cl_mem), (void *)&cl_mem_vI);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 19, sizeof(cl_mem), (void *)&cl_mem_refrac_countE);
	CHECK_STATUS(clStatus);
	clStatus = clSetKernelArg(kernel, 20, sizeof(cl_mem), (void *)&cl_mem_refrac_countI);
	CHECK_STATUS(clStatus);	
	
	//=====================================================
	// Copy the Buffers to the device
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_theta, CL_TRUE, 0, NUM_NEURONS * sizeof(float), theta, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_weightsXeAe, CL_TRUE, 0, (784*NUM_NEURONS) * sizeof(float), weightsXeAe, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_weightsAeAi, CL_TRUE, 0, (NUM_NEURONS*NUM_NEURONS) * sizeof(float), weightsAeAi, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_weightsAiAe, CL_TRUE, 0, (NUM_NEURONS*NUM_NEURONS) * sizeof(float), weightsAiAe, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_input_sample, CL_TRUE, 0, (single_sample_time*numPixels)* sizeof(bool), input_sample, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikesXePre, CL_TRUE, 0, numPixels * sizeof(bool), spikesXePre, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikesXePos, CL_TRUE, 0, numPixels * sizeof(bool), spikesXePos, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ae_Ai_pre, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ae_Ai_pre, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ae_Ai_pos, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ae_Ai_pos, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ai_Ae_pre, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ai_Ae_pre, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ai_Ae_pos, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ai_Ae_pos, 0, NULL, NULL);
	CHECK_STATUS(clStatus);	
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spike_count, CL_TRUE, 0, NUM_NEURONS * sizeof(unsigned short int), spike_count, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_assignments, CL_TRUE, 0, NUM_NEURONS * sizeof(unsigned short int), assignments, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_digits, CL_TRUE, 0, (10) * sizeof(unsigned short int), digits, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_vE, CL_TRUE, 0, NUM_NEURONS * sizeof(float), vE, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_vI, CL_TRUE, 0, NUM_NEURONS * sizeof(float), vI, 0, NULL, NULL);
	CHECK_STATUS(clStatus);
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_refrac_countE, CL_TRUE, 0, NUM_NEURONS * sizeof(int), refrac_countE, 0, NULL, NULL);
	CHECK_STATUS(clStatus);		
	clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_refrac_countI, CL_TRUE, 0, NUM_NEURONS * sizeof(int), refrac_countI, 0, NULL, NULL);
	CHECK_STATUS(clStatus);

    time_t start, end;
    time(&start);
    //! =====     =====     =====
    //! Run Simulation
    //! =====     =====     =====
    #if DEBUG_CASE != 0
    for (int numSample = 1; numSample <= 1; ++numSample) {
	#else
	for (int numSample = 1; numSample <= 10000; ++numSample) {
	#endif

        char buffer[100];
        sprintf( buffer, PATH_SAMPLES_POISSON ,numSample);

        std::string filename(buffer);

        getInputSample( input_sample, filename, 250, 784);
        
        clStatus = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&cl_mem_input_sample);
		CHECK_STATUS(clStatus);
		
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_input_sample, CL_TRUE, 0, (single_sample_time*numPixels)* sizeof(bool), input_sample, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		
        //! Simulate network activity for 250 timesteps.
        #if DEBUG_CASE != 0
        for (int indT = 248; indT < 250; ++indT)
        #else
        for (int indT = 0; indT < single_sample_time; ++indT)
        #endif
        {	
			clStatus = clSetKernelArg(kernel, 1, sizeof(int), (void *)&indT);
			CHECK_STATUS(clStatus);
			
            size_t global_size = NUM_NEURONS; // Process the entire lists
			size_t local_size = NUM_NEURONS/2;   // Process one item at a time
			clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
			CHECK_STATUS(clStatus);
			
			// Read the cl memory C_clmem on device to the host variables
			#if DEBUG_CASE == 1
			clStatus = clEnqueueReadBuffer(command_queue, cl_mem_refrac_countE, CL_TRUE, 0, NUM_NEURONS * sizeof(int), refrac_countE, 0, NULL, NULL);
			CHECK_STATUS(clStatus);
			
			for( int i = 0 ; i < 400; i++){
				if( input_sample[i+indT*784] != 0)
					printf("time[%03d][%d] = %d \n",indT ,i ,refrac_countE[i]);
			}
			
			#elif DEBUG_CASE == 2
			clStatus = clEnqueueReadBuffer(command_queue, cl_mem_spikesXePre, CL_TRUE, 0, numPixels * sizeof(bool), spikesXePre, 0, NULL, NULL);
			CHECK_STATUS(clStatus);
			
			for( int i = 0 ; i < 784; i++){
				if( spikesXePre[i] == 1)
					printf("time[%03d][%d] = %d \n",indT ,i ,spikesXePre[i]);
			}
			#else
			clStatus = clEnqueueReadBuffer(command_queue, cl_mem_spike_count, CL_TRUE, 0, NUM_NEURONS * sizeof(unsigned short int), spike_count, 0, NULL, NULL);
			CHECK_STATUS(clStatus);
			#endif
        }

        std::cout << "Digit class: " << classification(spike_count,assignments) << std::endl;

        std::ofstream fileLabels;
        std::string filenameLabels = std::string(PATH_RESULTS_NET) + "labelsVIM3Paralelo.csv";
        fileLabels.open(filenameLabels, std::ofstream::out | std::ofstream::app);
        if (!fileLabels.is_open())
        {
            std::cout << "Error opening file labelsVIM3Paralelo.csv" << __LINE__ << std::endl;
            exit(1);
        }

        fileLabels << classification(spike_count, assignments) << std::endl;
        fileLabels.close();

        //! =====     =====     =====
        //! Reset Variables
        //! =====     =====     =====
        for (int i = 0; i < NUM_NEURONS; ++i) {

            vE[i] = v_rest_e;
            vI[i] = v_rest_i;

            refrac_countE[i] = 0;
            refrac_countI[i] = 0;

            spikes_Ae_Ai_pos[i] = 0;
            spikes_Ae_Ai_pre[i] = 0;
            spikes_Ai_Ae_pos[i] = 0;
            spikes_Ai_Ae_pre[i] = 0;
            
            spike_count[i] = 0;
        }

        for (int i = 0; i < 784; ++i) {
            spikesXePre[i] = 0;
            spikesXePos[i] = 0;
        }
        
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_vE, CL_TRUE, 0, NUM_NEURONS * sizeof(float), vE, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_vI, CL_TRUE, 0, NUM_NEURONS * sizeof(float), vI, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_refrac_countE, CL_TRUE, 0, NUM_NEURONS * sizeof(int), refrac_countE, 0, NULL, NULL);
		CHECK_STATUS(clStatus);		
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_refrac_countI, CL_TRUE, 0, NUM_NEURONS * sizeof(int), refrac_countI, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
        
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ae_Ai_pre, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ae_Ai_pre, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ae_Ai_pos, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ae_Ai_pos, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ai_Ae_pre, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ai_Ae_pre, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikes_Ai_Ae_pos, CL_TRUE, 0, NUM_NEURONS * sizeof(bool), spikes_Ai_Ae_pos, 0, NULL, NULL);
		CHECK_STATUS(clStatus);	
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spike_count, CL_TRUE, 0, NUM_NEURONS * sizeof(unsigned short int), spike_count, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikesXePre, CL_TRUE, 0, numPixels * sizeof(bool), spikesXePre, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
		clStatus = clEnqueueWriteBuffer(command_queue, cl_mem_spikesXePos, CL_TRUE, 0, numPixels * sizeof(bool), spikesXePos, 0, NULL, NULL);
		CHECK_STATUS(clStatus);
    }

    // Recording end time.
    time(&end);

    // Calculating total time taken by the program.
    double time_taken = double(end - start);
    cout << "Time taken by program is : " << time_taken << " sec" << endl;
    
    
    // Finally release all OpenCL allocated objects and host buffers.
	clStatus = clReleaseKernel(kernel);
	clStatus = clReleaseProgram(program);
	
	
	clStatus = clReleaseMemObject( cl_mem_spikes_Ai_Ae_pos );
	CHECK_STATUS(clStatus);
	clStatus = clReleaseMemObject( cl_mem_weightsAiAe );
	CHECK_STATUS(clStatus);
	clStatus = clReleaseMemObject( cl_mem_spikesXePos );
	CHECK_STATUS(clStatus);
	clStatus = clReleaseMemObject( cl_mem_weightsXeAe );
	CHECK_STATUS(clStatus);
	clStatus = clReleaseMemObject( cl_mem_vE );
	CHECK_STATUS(clStatus);
	clStatus = clReleaseMemObject( cl_mem_refrac_countE );
	CHECK_STATUS(clStatus);
	clStatus = clReleaseMemObject( cl_mem_spikes_Ae_Ai_pre );
	CHECK_STATUS(clStatus);
	clStatus = clReleaseMemObject( cl_mem_theta );
	CHECK_STATUS(clStatus);

	delete [] vE;
	delete [] vI;
	
	delete [] weightsXeAe;
	delete [] weightsAeAi;
	delete [] weightsAiAe;
	
	delete [] spikesXePre;
	delete [] spikesXePos;
	
	delete [] refrac_countE;
	delete [] refrac_countI;
	
	delete [] input_sample;
	delete [] spike_count;
	
	delete [] theta;
	delete [] assignments;
	delete [] proportions;
	
	clStatus = clReleaseCommandQueue(command_queue);
	CHECK_STATUS(clStatus);
	clStatus = clReleaseContext(context);
	CHECK_STATUS(clStatus);
	free(platforms);
	free(device_list);
	
	return 0;
}
