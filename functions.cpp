#include "functions.h"

void getAssignments(unsigned short int *assignments , const std::string fileName)
{
    std::string data;

    std::ifstream infile;
    infile.open( fileName );

    if (!infile.is_open())
    {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    std::cout << "Read file" << std::endl;

    infile >> data;

    if (infile.eof())
    {
        std::cout << "End of file " << fileName << " reached successfully" << std::endl;
    }
    int n1 = 0, n2 = 0, longChar = 0;
    for (int indxj = 0; indxj < 400; ++indxj) {
        n2 = data.find(",", n1);
        longChar = n2 - n1;
        if (n2 == std::string::npos) {
            //cout << "not found\n";
            n2 = data.length();
            longChar = n2 - n1;

            //cout << "found: " << s.substr(n1,longChar) << endl;
            assignments[ indxj ] = std::stoi(data.substr(n1,longChar));
            //cout << "found: " << weights[indx] << endl;
        } else {
            //cout << "found: " << s.substr(n1,longChar) << endl;
            assignments[ indxj ] = std::stoi(data.substr(n1,longChar));
            //cout << "found: " << weights[indx] << endl;
        }
        n1 = n2 + 1;
    }

    infile.close();
}

void getProportions(float *proportions , const int rows, const int cols, const std::string fileName)
{
    std::string data;

    std::ifstream infile;
    infile.open( fileName );

    if (!infile.is_open())
    {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    std::cout << "Read file" << std::endl;

    infile >> data;

    if (infile.eof())
    {
        std::cout << "End of file " << fileName << " reached successfully" << std::endl;
    }
    int n1 = 0, n2 = 0, longChar = 0;
    for (int indxj = 0; indxj < rows*cols; ++indxj) {
        n2 = data.find(",", n1);
        longChar = n2 - n1;
        if (n2 == std::string::npos) {
            //cout << "not found\n";
            n2 = data.length();
            longChar = n2 - n1;

            //cout << "found: " << s.substr(n1,longChar) << endl;
            proportions[ indxj ] = std::stod(data.substr(n1,longChar));
            //cout << "found: " << weights[indx] << endl;
        } else {
            //cout << "found: " << s.substr(n1,longChar) << endl;
            proportions[ indxj ] = std::stod(data.substr(n1,longChar));
            //cout << "found: " << weights[indx] << endl;
        }
        n1 = n2 + 1;
    }

    infile.close();
}

void getWeights( float *weights, const std::string fileName, const int row, const int col )
{
    std::string data;

    std::ifstream infile;
    infile.open( fileName );

    if (!infile.is_open())
    {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    std::cout << "Read file" << std::endl;

    infile >> data;
    if (infile.eof())
    {
        std::cout << "End of file " << fileName <<  " reached successfully" << std::endl;
    }

    int n1 = 0, n2 = 0, longChar = 0;
    for (int indx = 0; indx < row*col; ++indx) {
        n2 = data.find(",", n1);
        longChar = n2 - n1;
        try {

            if (n2 == std::string::npos) {
                //cout << "not found\n";
                n2 = data.length();
                longChar = n2 - n1;

                //cout << "found: " << s.substr(n1,longChar) << endl;
                weights[ indx ] = static_cast<float>( std::stod(data.substr(n1,longChar)) );
                //cout << "found: " << weights[indx] << endl;
            } else {

                //cout << "found: " << s.substr(n1,longChar) << endl;
                std::string tmp = data.substr(n1,longChar);
                weights[ indx ] = static_cast<float>(std::stod(tmp));
                //cout << "found: " << weights[indx] << endl;
            }
            n1 = n2 + 1;
        } catch (std::exception& e)
        {
            std::cerr << "Standard exception: (" << e.what() << ") indx (" << indx << ") - LINE -> " << __LINE__ << " - FUNC -> " << __func__ << std::endl;
            exit(1);
        }
    }

    infile.close();
}

void getTheta( float *theta, const std::string fileName )
{
    std::string data;

    std::ifstream infile;
    infile.open( fileName );

    if (!infile.is_open())
    {
        std::cout << "Error opening file: " << __FILE__ << ", " << __FUNCTION__ << ", line: " << __LINE__ << std::endl;
        exit(1);
    }

    std::cout << "Read file" << std::endl;

    infile >> data;

    if (infile.eof())
    {
        std::cout << "End of file " << fileName << " reached successfully" << std::endl;
    }
    int n1 = 0, n2 = 0, longChar = 0;
    for (int indxj = 0; indxj < 400; ++indxj) {
        n2 = data.find(",", n1);
        longChar = n2 - n1;
        if (n2 == std::string::npos) {
            //cout << "not found\n";
            n2 = data.length();
            longChar = n2 - n1;

            //cout << "found: " << s.substr(n1,longChar) << endl;
            std::string tmp = data.substr(n1,longChar);
            theta[ indxj ] = std::stod(tmp);
            //cout << "found: " << weights[indx] << endl;
        } else {
            //cout << "found: " << s.substr(n1,longChar) << endl;
            std::string tmp = data.substr(n1,longChar);
            theta[ indxj ] = std::stod(tmp);
            //std::cout << theta[ indxj ] << std::endl;
            //cout << "found: " << weights[indx] << endl;
        }
        n1 = n2 + 1;
    }

    infile.close();
}

void getInputSample( bool *arr, const std::string fileName, const int row, const int col )
{
    std::string data;

    std::ifstream infile;
    infile.open( fileName );

    if (!infile.is_open())
    {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    std::cout << "Read file" << std::endl;

    infile >> data;
    if (infile.eof())
    {
        std::cout << "End of file " << fileName <<  " reached successfully" << std::endl;
    }

    int n1 = 0, n2 = 0, longChar = 0;
    for (int indx = 0; indx < row*col; ++indx) {
        n2 = data.find(",", n1);
        longChar = n2 - n1;
        try {

            if (n2 == std::string::npos) {
                //cout << "not found\n";
                n2 = data.length();
                longChar = n2 - n1;

                //cout << "found: " << s.substr(n1,longChar) << endl;
                arr[ indx ] = static_cast<bool>(std::stoi(data.substr(n1,longChar)) );
                //cout << "found: " << weights[indx] << endl;
            } else {

                //cout << "found: " << s.substr(n1,longChar) << endl;
                std::string tmp = data.substr(n1,longChar);
                arr[ indx ] = static_cast<bool>(std::stoi(tmp));
                //cout << "found: " << weights[indx] << endl;
            }
            n1 = n2 + 1;
        } catch (std::exception& e)
        {
            std::cerr << "Standard exception: (" << e.what() << ") indx (" << indx << ") - LINE -> " << __LINE__ << " - FUNC -> " << __func__ << std::endl;
            exit(1);
        }
    }

    infile.close();
}

float dotPointInputs(float *syn, bool *pixels_x_time, unsigned int neuronIndex ){
    float acum = 0.0f;

    for (int i = 0; i < 784; ++i) {
        acum += pixels_x_time[i] ? syn[i*400+neuronIndex] : 0.0f ;
    }
    return acum;
}

float dotPointLayers(float *syn, bool *spikeVector ){
    float acum = 0.0f;

    for (unsigned int i = 0; i < 400; ++i) {
        acum += spikeVector[i] ? syn[i] : 0.0f ;
    }
    return acum;
}


/*!
 * \brief This function verifies that at least one neuron performs a firing.
 * \param spikesE It is a vector of 1s or 0s
 * \return The index of the winning neuron, or -1 if there is no firing.
 */
int winner( bool *spikesE)
{
    // initialize random seed
    //srand (time_t(NULL));

    for (int indx = 0; indx < 400; ++indx) {
        if(spikesE[indx] != 0)
            return indx;
    }


    // Generate random number between 0 and 399
    /*
    do{
        ind = rand() % 400;

    }while( spikesE[ind] == 0 );
    */

    return -1;
}

void setOneSpike(bool *spikes, int indexWin, unsigned short int *digits, unsigned short int *assignments)
{
    /*
    std::ofstream file;

    std::string filename = _PATH + "inputSamples/vectorIndexresultsQt.csv";

    file.open (filename, std::ofstream::out | std::ofstream::app);

    if (!file.is_open())
    {
        std::cout << "Error opening file" << __LINE__ << std::endl;
        exit(1);
    }
    */

    static int countinput = 1;
    for (int index = 0; index < 400; ++index) {
        if(spikes[index]){
            //std::cout << "index [" << countinput << "] : " << index << " Digit [" << (unsigned short int)assignments[index] << "] " << std::endl;
            //std::cout << index << ",";
            //file << index << ",";
            ++digits[ assignments[index] ];
        }
        spikes[index] = (index == indexWin) ? 1 : 0;
    }
    countinput++;

    //file.close();
}

void resetInputs(bool *spikesE, bool *spikesI)
{
    for (int index = 0; index < 400; ++index) {
        spikesE[index] = 0;
        spikesI[index] = 0;
    }
}

int classification(unsigned short int *spike_count, unsigned short int *assignments){

    float rates[10] = {0.0f};

    //! Count the number of neurons with this label assignment.
    float n_assigns[10] = {0};

    for (int indx = 0; indx < 400; ++indx) {
        ++n_assigns[assignments[indx]];

        rates[assignments[indx]] += spike_count[indx];
    }

    for (int indx = 0; indx < 10; ++indx) {
        rates[indx] /= n_assigns[indx];
    }

    int indWinner = 0;
    float ratesWin = 0;

    for (int indx = 0; indx < 10; ++indx) {
        if ( rates[indx] > ratesWin ) {
            indWinner = indx;
            ratesWin = rates[indx];
        }
    }
    return indWinner;
}
