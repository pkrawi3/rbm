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

        // Constructor for RBM
        RBM( vector<int> vis, vector<int> hid)
        {
            v = vis;
            h = hid;
            W.resize(v.size());
            a.resize(v.size());
            b.resize(h.size());

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
                    first = first + v[i] * W[i][j] * h[j];
                }
            }

            // Second portion
            double second = 0;
            for(int i=0; i<v.size(); i++)
            {
                second = second + v[i] * a[i];
            }

            // Third portion
            double third = 0;
            for(int j=0; j<h.size(); j++)
            {
                third = third + h[j] * b[j];
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
                double m = 0;
                for(int j=0; j<h.size(); j++)
                {
                    m += W[i][j] * h[j] + a[i];
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
                double m = 0;
                for(int i=0; i<v.size(); i++)
                {
                    m += W[i][j] * v[i] + b[j];
                }
                double prob =  exp(m) / ( exp(m) + exp(-1*m) );
                if(unif(mt) < prob) h[j] = 1;
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


void RBM_h_run()
{
    // Testing Portion 1
    vector<int> initial_v = generate_random_states(5);
    vector<int> initial_h = generate_random_states(2);

    RBM cur_net(initial_v, initial_h);

    // Generate Histogram
    map<string, int> probs;
    int norm = 0;
    for(int i=0; i<100000; i++)
    {
        cur_net.sample_h();
        string cur = cur_net.to_string(cur_net.h);
        if(probs.find(cur) == probs.end()) probs.insert(make_pair(cur, 0));
        probs[cur] += 1;
        norm += 1;
    }

    // Write the map to a file
    ofstream cur_file;
    cur_file.open("h_sample.txt");
    cur_file << cur_net.to_string(initial_v) << endl;
    for (map<string,int>::iterator it=probs.begin(); it!=probs.end(); ++it)
    {
        cur_file << it->first << "," << double(it->second) / 100000.0 << '\n';
    }

    // Generate Brute-Force Calulation
    double Z = 0;
    vector<double> h_vec(4);
    for(int i=0; i<4; i++)
    {
        cur_net.h = cur_net.from_string(gen_str_2(i));
        double cur = exp(-1*cur_net.calc_energy());
        h_vec[i] = cur; 
        Z += cur;
    }
    for(int i=0; i<4; i++)
    {
        cur_file << h_vec[i] / Z << endl;
    }


    cur_file.close();
}

void RBM_v_run()
{
    // Testing Portion 1
    vector<int> initial_v = generate_random_states(5);
    vector<int> initial_h = generate_random_states(2);

    RBM cur_net(initial_v, initial_h);

    // Generate Histogram
    map<string, int> probs;
    int norm = 0;
    for(int i=0; i<1000000; i++)
    {
        cur_net.sample_v();
        string cur = cur_net.to_string(cur_net.v);
        if(probs.find(cur) == probs.end()) probs.insert(make_pair(cur, 0));
        probs[cur] += 1;
        norm += 1;
    }

    // Write the map to a file
    ofstream cur_file;
    cur_file.open("v_sample.txt");
    cur_file << cur_net.to_string(initial_h) << endl;
    for (map<string,int>::iterator it=probs.begin(); it!=probs.end(); ++it)
    {
        cur_file << it->first << "," << double(it->second) / 1000000.0 << '\n';
    }

    // Generate Brute-Force Calulation
    double Z = 0;
    vector<double> v_vec(32);
    for(int i=0; i<32; i++)
    {
        cur_net.v = cur_net.from_string(gen_str_5(i));
        double cur = exp(-1*cur_net.calc_energy());
        v_vec[i] = cur; 
        Z += cur;
    }
    for(int i=0; i<32; i++)
    {
        cur_file << v_vec[i] / Z << endl;
    }

    cur_file.close();
}

void RBM_third_run()
{
    // Testing Portion 1
    vector<int> initial_v = generate_random_states(5);
    vector<int> initial_h = generate_random_states(2);

    RBM cur_net(initial_v, initial_h);

    // Generate Histogram
    map<string, int> probs;
    int norm = 0;
    for(int i=0; i<10000000; i++)
    {
        // 10 Runs of Gibbs
        for(int j=0; j<10; j++)
        {
            cur_net.sample_h();
            cur_net.sample_v();
        }
        string cur = cur_net.to_string(cur_net.v);
        cur += "_";
        cur += cur_net.to_string(cur_net.h);
        if(probs.find(cur) == probs.end()) probs.insert(make_pair(cur, 0));
        probs[cur] += 1;
        norm += 1;
    }

    // Write the map to a file
    ofstream cur_file;
    cur_file.open("run_three.txt");
    //cur_file << cur_net.to_string(initial_h) << endl;
    for (map<string,int>::iterator it=probs.begin(); it!=probs.end(); ++it)
    {
        cur_file <<  it->first << ':'<< double(it->second) / 10000000.0 << '\n';
    }

    double Z = 0;
    vector<double> v_vec(32*4, 0);

    for(int i=0; i<32; i++)
    {
        for(int j=0; j<4; j++)
        {
            cur_net.v = cur_net.from_string(gen_str_5(i));
            cur_net.h = cur_net.from_string(gen_str_2(j));
            double cur = exp(-1*cur_net.calc_energy());
            //cout << i + j << endl;
            v_vec[i + j*32] = cur; 
            Z += cur;
        }
    }
    for(int i=0; i<32*4; i++)
    {
        cur_file << v_vec[i] / Z << endl;
    }

    cur_file.close();

}

void RBM_pv()
{
    // Testing Portion 1
    vector<int> initial_v = generate_random_states(5);
    vector<int> initial_h = generate_random_states(2);

    RBM cur_net(initial_v, initial_h);

    // Generate Histogram
    map<string, int> probs;
    int norm = 0;
    for(int i=0; i<1000000; i++)
    {
        cur_net.sample_v();
        string cur = cur_net.to_string(cur_net.v);
        if(probs.find(cur) == probs.end()) probs.insert(make_pair(cur, 0));
        probs[cur] += 1;
        norm += 1;
    }

    // Write the map to a file
    ofstream cur_file;
    cur_file.open("v_sample.txt");
    cur_file << cur_net.to_string(initial_h) << endl;
    for (map<string,int>::iterator it=probs.begin(); it!=probs.end(); ++it)
    {
        cur_file << it->first << "," << double(it->second) / 1000000.0 << '\n';
    }

    // Generate Brute-Force Calulation
    double Z = 0;
    vector<double> v_vec(32);
    for(int i=0; i<32; i++)
    {
        cur_net.v = cur_net.from_string(gen_str_5(i));
        double cur = exp(-1*cur_net.calc_energy());
        v_vec[i] = cur; 
        Z += cur;
    }
    for(int i=0; i<32; i++)
    {
        cur_file << v_vec[i] / Z << endl;
    }

    cur_file.close(); 
}


void gen_theory()
{
    RBM cur_net(generate_random_states(5), generate_random_states(2));

    // Generate p v h
    ofstream cur_file;
    cur_file.open("pvh_theory.txt");

    double Z = 0;
    vector<double> v_vec(32*4);

    for(int i=0; i<32; i++)
    {
        for(int j=0; j<4; i++)
        {
            cur_net.v = cur_net.from_string(gen_str_5(i));
            cur_net.h = cur_net.from_string(gen_str_2(j));
            double cur = exp(-1*cur_net.calc_energy());
            v_vec[i + j*32] = cur; 
            Z += cur;
        }
    }
    for(int i=0; i<32*4; i++)
    {
        cur_file << v_vec[i] / Z << endl;
    }

    cur_file.close();


    // // Generate p v
    // ofstream cur_file;
    // cur_file.open("pv_theory.txt");

    // double Z = 0;
    // vector<double> v_vec(32*4);

    // for(int i=0; i<32; i++)
    // {
    //     for(int j=0; j<4; i++)
    //     {
    //         cur_net.v = cur_net.from_string(gen_str_5(i));
    //         cur_net.h = curr_net.from_string(gen_str_2(j));
    //         double cur = exp(-1*cur_net.calc_energy());
    //         v_vec[i + j*32] = cur; 
    //         Z += cur;
    //     }
    // }
    // for(int i=0; i<32*4; i++)
    // {
    //     cur_file << v_vec[i] / Z << endl;
    // }

    // cur_file.close();
    
    
    // // Generate p h 
    // ofstream cur_file;
    // cur_file.open("ph_theory.txt");


    cur_file.close();
}



// Main Function
int main(){
    cout << "My RBM Program" << endl;

    random_device rd;
    mt = mt19937(rd());    

    RBM_third_run();

    cout << "My RGM Program Finished" << endl;
    return 0;
}