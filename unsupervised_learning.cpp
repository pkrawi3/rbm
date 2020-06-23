// Class File for RBM Implementation

#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <map>
#include <fstream>
#include <bitset>

using namespace std;

// Globals
mt19937 mt;

// RBM Class
class RBM{

    // Public Members
    public:
        vector <vector <double> > W;    // Matrix of weights
        vector <int> v;                 // Vector of visible neurons
        vector <int> h;                 // Vector of hidden neurons
        vector <double> a;              // Vector for visible biases
        vector <double> b;              // Vector for hidden biases
        double gradient_W;
        double gradient_v;
        double gradient_h;

        // Constructor for RBM
        RBM( vector<int> vis, vector<int> hid)
        {
            v = vis;
            h = hid;
            W.resize(v.size());
            a.resize(v.size());
            b.resize(h.size());
            gradient_W = 0.0;
            gradient_v = 0.0;
            gradient_h = 0.0;

            // Random distribution
            uniform_real_distribution <double> unif(-1.0, 1.0);
    
            // Randomize initial vectors
            for(int i=0; i<v.size(); i++)
            {
                W[i].resize(h.size()); 
                for(int j=0; j<h.size(); j++)
                {
                    W[i][j] = unif(mt);
                }
                a[i] = unif(mt);
            }
            for(int j=0; j<h.size(); j++)
            {
                b[j] = unif(mt);
            }
        }

        // Computes the Energy
        double calc_energy()
        {
            // First portion
            double first = 0;
            for(int i=0; i<v.size(); i++)
            {
                for(int j=0; j<h.size(); j++)
                {
                   // cout << W[i][j] << '\n';
                    first = first + double(v[i]) * W[i][j] * h[j];
                }
            }

            // Second portion
            double second = 0;
            for(int i=0; i<v.size(); i++)
            {
                second = second + double(v[i]) * a[i];
            }

            // Third portion
            double third = 0;
            for(int j=0; j<h.size(); j++)
            {
                third = third + double(h[j]) * b[j];
            }

            return -1*first - second - third;
        }

        // Basic sampling function for updating visible layer
        void sample_v()
        {
            // Random distribution
            uniform_real_distribution <double> unif(0.0, 1.0);

            // Sample for each neuron
            for(int i=0; i<v.size(); i++)
            {
                // Computer m
                double m = a[i];
                for(int j=0; j<h.size(); j++)
                {
                    m += W[i][j] * h[j];
                }
                double prob =  exp(m) / ( exp(m) + exp(-1*m) );

                if(unif(mt) < prob) v[i] = 1;
                else v[i] = -1;
            }
            return;
        }

        // Basic sampling function for updating the hidden layer
        void sample_h()
        {
            // Random distribution
            uniform_real_distribution <double> unif(0.0, 1.0);

            // Sample for each neuron
            for(int j=0; j<h.size(); j++)
            {
                // Computer m
                double m = b[j];
                for(int i=0; i<v.size(); i++)
                {
                    m += W[i][j] * v[i];
                }
                double prob =  exp(m) / ( exp(m) + exp(-1*m) );
                double cur_p = unif(mt);
                if(cur_p < prob) h[j] = 1;
                else h[j] = -1;
            }
            return;
        }

        string to_string(vector<int> temp)
        {
            string result = "";
            for(int i=0; i<temp.size(); i++)
            {
                if(temp[i] > 0) result += '1';
                else result += '0';
            }
            return result;
        }

        vector<int> from_string(string temp)
        {
            vector<int> result;
            for(int i=0; i<temp.size(); i++)
            {
                if(temp[i] == '0') result.push_back(-1);
                else result.push_back(1);
            }
            return result;
        }

};

// Generates a randomized initial vector of neuron states
vector<int> generate_random_states(int size)
{
    // Random distribution
    uniform_int_distribution <int> unif(0, 1);
    
    vector <int> temp(size, 0);
    for(int i=0; i<size; i++)
    {
        if(unif(mt) > 0) temp[i] = 1;
        else temp[i] = -1;
    }
    return temp;
}

string gen_str_5(int n)
{
    string result = bitset <5> (n).to_string(); 
    return result;
}

string gen_str_2(int n)
{
    string result = bitset <2> (n).to_string(); 
    return result;
}

int main()
{

    return 0;
}